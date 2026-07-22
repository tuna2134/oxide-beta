use super::{BertForSequenceClassification, BertInput};
use oxide_torch::loss::cross_entropy;
use oxide_torch::nn::{Module, Trainable};
use oxide_torch::optim::{AdamW, Optimizer};
use oxide_torch::{Device, Tensor};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, serialize_to_file};
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn full_encoder_receives_gradients_and_updates_on_cpu() {
    let directory = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(directory.path());
    let mut model =
        BertForSequenceClassification::from_pretrained(directory.path(), Device::Cpu).unwrap();
    let input = BertInput::from_ids(
        &[vec![1, 2, 3], vec![1, 4, 5]],
        Some(&[vec![1, 1, 1], vec![1, 1, 1]]),
        Some(&[vec![0, 0, 0], vec![0, 0, 0]]),
    )
    .unwrap();
    let labels = Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap();

    let logits = model.forward(&input).unwrap();
    assert_eq!(logits.shape(), &[2, 2]);
    assert!(
        logits
            .to_vec()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    let embedding_before = first_parameter_values(&model);
    let loss = cross_entropy(&logits, &labels).unwrap();
    loss.backward().unwrap();

    let mut parameter_count = 0;
    let mut gradient_count = 0;
    let mut encoder_embedding_has_gradient = false;
    model.visit_parameters(&mut |parameter| {
        parameter_count += 1;
        if let Some(gradient) = parameter.value().grad().unwrap() {
            gradient_count += 1;
            if parameter_count == 1 {
                encoder_embedding_has_gradient = gradient.iter().any(|value| value.abs() > 1.0e-12);
            }
        }
    });
    assert_eq!(gradient_count, parameter_count);
    assert!(encoder_embedding_has_gradient);

    let mut optimizer = AdamW::new(0.01, 0.0).unwrap();
    optimizer.step(&mut model).unwrap();
    let embedding_after = first_parameter_values(&model);
    assert_ne!(embedding_before, embedding_after);

    for _ in 0..4 {
        optimizer.zero_grad(&model).unwrap();
        let loss = cross_entropy(&model.forward(&input).unwrap(), &labels).unwrap();
        loss.backward().unwrap();
        optimizer.step(&mut model).unwrap();
    }
    let final_loss = cross_entropy(&model.forward(&input).unwrap(), &labels)
        .unwrap()
        .item()
        .unwrap();
    assert!(final_loss.is_finite());
}

#[test]
fn encoder_runs_through_the_shared_cpu_jit() {
    let directory = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(directory.path());
    let model =
        BertForSequenceClassification::from_pretrained(directory.path(), Device::Cpu).unwrap();
    let input = BertInput::from_ids(
        &[vec![1, 2, 3]],
        Some(&[vec![1, 1, 1]]),
        Some(&[vec![0, 0, 0]]),
    )
    .unwrap();
    let examples = [
        input.input_ids.clone(),
        input.attention_mask.clone().unwrap(),
        input.token_type_ids.clone().unwrap(),
    ];
    let compiled = oxide_torch::jit::trace(&examples, |values| {
        model.forward(&BertInput {
            input_ids: values[0].clone(),
            attention_mask: Some(values[1].clone()),
            token_type_ids: Some(values[2].clone()),
        })
    })
    .unwrap();
    let expected = model.forward(&input).unwrap().to_vec().unwrap();
    let first = compiled.run(&examples).unwrap().to_vec().unwrap();
    let second = compiled.run(&examples).unwrap().to_vec().unwrap();
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert_eq!(compiled.cached_specializations(), 1);
}

#[test]
fn legacy_layer_norm_names_are_supported() {
    let directory = tempfile::tempdir().unwrap();
    write_tiny_checkpoint_with_options(directory.path(), true, true);
    BertForSequenceClassification::from_pretrained(directory.path(), Device::Cpu).unwrap();
}

#[test]
fn missing_classification_head_is_initialized() {
    let directory = tempfile::tempdir().unwrap();
    write_tiny_checkpoint_with_options(directory.path(), false, false);
    let model =
        BertForSequenceClassification::from_pretrained(directory.path(), Device::Cpu).unwrap();
    let mut shapes = Vec::new();
    model.visit_parameters(&mut |parameter| shapes.push(parameter.value().shape().to_vec()));
    assert_eq!(&shapes[shapes.len() - 2..], &[vec![2, 4], vec![2]]);
}

#[test]
fn saved_classification_checkpoint_round_trips() {
    let source = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(source.path());
    let model = BertForSequenceClassification::from_pretrained(source.path(), Device::Cpu).unwrap();
    let input = BertInput::from_ids(&[vec![1, 2, 3]], None, None).unwrap();
    let expected = model.forward(&input).unwrap().to_vec().unwrap();

    model.save_pretrained(output.path()).unwrap();
    let reloaded =
        BertForSequenceClassification::from_pretrained(output.path(), Device::Cpu).unwrap();
    assert_eq!(
        reloaded.forward(&input).unwrap().to_vec().unwrap(),
        expected
    );
}

fn first_parameter_values(model: &BertForSequenceClassification) -> Vec<f32> {
    let mut result = None;
    model.visit_parameters(&mut |parameter| {
        if result.is_none() {
            result = Some(parameter.value().to_vec().unwrap());
        }
    });
    result.unwrap()
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn write_tiny_checkpoint(directory: &Path) {
    write_tiny_checkpoint_with_options(directory, false, true);
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn write_tiny_checkpoint_with_options(directory: &Path, legacy: bool, classifier: bool) {
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    let mut insert = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
        let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
        tensors.insert(name.into(), (shape, bytes));
    };
    let values = |len: usize| {
        (0..len)
            .map(|index| ((index * 13 % 17) as f32 - 8.0) * 0.01)
            .collect()
    };
    insert(
        "bert.embeddings.word_embeddings.weight",
        vec![8, 4],
        values(32),
    );
    insert(
        "bert.embeddings.position_embeddings.weight",
        vec![8, 4],
        values(32),
    );
    insert(
        "bert.embeddings.token_type_embeddings.weight",
        vec![2, 4],
        values(8),
    );
    let norm_weight = if legacy { "gamma" } else { "weight" };
    let norm_bias = if legacy { "beta" } else { "bias" };
    insert(
        &format!("bert.embeddings.LayerNorm.{norm_weight}"),
        vec![4],
        vec![1.0; 4],
    );
    insert(
        &format!("bert.embeddings.LayerNorm.{norm_bias}"),
        vec![4],
        vec![0.0; 4],
    );
    for name in ["query", "key", "value"] {
        insert(
            &format!("bert.encoder.layer.0.attention.self.{name}.weight"),
            vec![4, 4],
            values(16),
        );
        insert(
            &format!("bert.encoder.layer.0.attention.self.{name}.bias"),
            vec![4],
            vec![0.0; 4],
        );
    }
    for (name, shape, data) in [
        (
            "bert.encoder.layer.0.attention.output.dense.weight",
            vec![4, 4],
            values(16),
        ),
        (
            "bert.encoder.layer.0.attention.output.dense.bias",
            vec![4],
            vec![0.0; 4],
        ),
        (
            &format!("bert.encoder.layer.0.attention.output.LayerNorm.{norm_weight}"),
            vec![4],
            vec![1.0; 4],
        ),
        (
            &format!("bert.encoder.layer.0.attention.output.LayerNorm.{norm_bias}"),
            vec![4],
            vec![0.0; 4],
        ),
        (
            "bert.encoder.layer.0.intermediate.dense.weight",
            vec![8, 4],
            values(32),
        ),
        (
            "bert.encoder.layer.0.intermediate.dense.bias",
            vec![8],
            vec![0.0; 8],
        ),
        (
            "bert.encoder.layer.0.output.dense.weight",
            vec![4, 8],
            values(32),
        ),
        (
            "bert.encoder.layer.0.output.dense.bias",
            vec![4],
            vec![0.0; 4],
        ),
        (
            &format!("bert.encoder.layer.0.output.LayerNorm.{norm_weight}"),
            vec![4],
            vec![1.0; 4],
        ),
        (
            &format!("bert.encoder.layer.0.output.LayerNorm.{norm_bias}"),
            vec![4],
            vec![0.0; 4],
        ),
        ("bert.pooler.dense.weight", vec![4, 4], values(16)),
        ("bert.pooler.dense.bias", vec![4], vec![0.0; 4]),
    ] {
        insert(name, shape, data);
    }
    if classifier {
        insert("classifier.weight", vec![2, 4], values(8));
        insert("classifier.bias", vec![2], vec![0.0; 2]);
    }
    let views = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &directory.join("model.safetensors")).unwrap();
    std::fs::write(
        directory.join("config.json"),
        r#"{"vocab_size":8,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":8,"max_position_embeddings":8,"type_vocab_size":2,"layer_norm_eps":1e-5,"num_labels":2}"#,
    )
    .unwrap();
}

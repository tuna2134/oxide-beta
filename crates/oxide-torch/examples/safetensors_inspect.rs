use oxide_torch::safetensors::SafeTensorLoader;

fn main() -> oxide_torch::Result<()> {
    let path = std::env::args_os().nth(1).ok_or_else(|| {
        oxide_torch::Error::Execution("usage: safetensors_inspect MODEL_DIR [FILTER]".into())
    })?;
    let filter = std::env::args().nth(2).unwrap_or_default();
    let loader = SafeTensorLoader::open(path)?;
    let mut names: Vec<_> = loader
        .names()
        .filter(|name| name.contains(&filter))
        .map(str::to_owned)
        .collect();
    names.sort();
    for name in names {
        let metadata = loader.metadata(&name)?;
        println!("{name}\t{:?}\t{:?}", metadata.dtype, metadata.shape);
    }
    Ok(())
}

use oxide_torch::{Device, Error, Result};

/// Removes a `--device` option and returns the requested execution device.
///
/// # Errors
///
/// Returns an error for malformed or unsupported device specifications, or
/// when CUDA was requested without compiling the `cuda` feature.
pub fn take_device(arguments: &mut Vec<String>) -> Result<Device> {
    let value = if let Some(index) = arguments.iter().position(|value| value == "--device") {
        if index + 1 == arguments.len() {
            return Err(Error::Execution(
                "--device requires cpu, cuda, or cuda:INDEX".into(),
            ));
        }
        let value = arguments.remove(index + 1);
        arguments.remove(index);
        Some(value)
    } else {
        arguments
            .iter()
            .position(|value| value.starts_with("--device="))
            .map(|index| arguments.remove(index)["--device=".len()..].to_owned())
    };

    match value.as_deref().unwrap_or("cpu") {
        "cpu" => Ok(Device::Cpu),
        "cuda" => cuda_device(0),
        value if value.starts_with("cuda:") => {
            let index = value["cuda:".len()..].parse::<usize>().map_err(|_| {
                Error::Execution(format!(
                    "invalid CUDA device `{value}`; expected cuda:INDEX"
                ))
            })?;
            cuda_device(index)
        }
        value => Err(Error::Execution(format!(
            "unsupported device `{value}`; expected cpu, cuda, or cuda:INDEX"
        ))),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::unnecessary_wraps)]
fn cuda_device(index: usize) -> Result<Device> {
    Ok(Device::Cuda(index))
}

#[cfg(not(feature = "cuda"))]
fn cuda_device(_index: usize) -> Result<Device> {
    Err(Error::Execution(
        "CUDA support is not compiled; rerun with `--features cuda`".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_and_rejects_unknown_devices() {
        let mut cpu = vec!["--device=cpu".into(), "text".into()];
        assert_eq!(take_device(&mut cpu).unwrap(), Device::Cpu);
        assert_eq!(cpu, ["text"]);

        let mut invalid = vec!["--device=tpu".into()];
        assert!(take_device(&mut invalid).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn parses_cuda_device_index() {
        let mut arguments = vec!["--device".into(), "cuda:2".into()];
        assert_eq!(take_device(&mut arguments).unwrap(), Device::Cuda(2));
        assert!(arguments.is_empty());
    }
}

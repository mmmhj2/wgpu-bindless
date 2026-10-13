/// Error type for asset I/O.
///
/// **Prototype stub.** The current implementation never constructs this; any
/// failure path simply `panic!`s. The type exists so function signatures can
/// stay stable until proper error handling is introduced.
#[derive(Debug)]
pub struct AssetIoError(pub String);

impl std::fmt::Display for AssetIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "asset I/O error: {}", self.0)
    }
}

impl std::error::Error for AssetIoError {}

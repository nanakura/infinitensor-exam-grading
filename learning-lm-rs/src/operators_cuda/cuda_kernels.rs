pub const MATMUL_TRANSB: &str = include_str!(concat!(env!("OUT_DIR"), "/matmul.ptx"));
pub const RMSNORM: &str = include_str!(concat!(env!("OUT_DIR"), "/rms_norm.ptx"));
pub const SWIGLU: &str = include_str!(concat!(env!("OUT_DIR"), "/swiglu.ptx"));
pub const ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));
pub const GATHER: &str = include_str!(concat!(env!("OUT_DIR"), "/gather.ptx"));

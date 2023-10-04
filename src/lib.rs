#![allow(non_snake_case)]

mod batch;
pub mod bbob;

// Re-exports.
pub use coco_futhark::Context;

pub mod backends {
    pub use coco_futhark::backends::Backend;
    #[cfg(feature = "cuda")]
    pub use coco_futhark::backends::Cuda;
    #[cfg(feature = "multicore")]
    pub use coco_futhark::backends::MultiCore;
    #[cfg(feature = "opencl")]
    pub use coco_futhark::backends::OpenCl;
    #[cfg(feature = "c")]
    pub use coco_futhark::backends::C;
}

pub use strum::IntoEnumIterator;

pub use crate::batch::InputBatch;

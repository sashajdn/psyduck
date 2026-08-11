//! CUDA kernel sources and their stable entry-point names.
//!
//! This crate owns the boundary between CUDA function names and Rust. Runtime
//! compilation, module loading, and launching belong to the backend consuming
//! these descriptors.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, DriverError},
    nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts},
};

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error(transparent)]
    DriverError(#[from] DriverError),
    #[error(transparent)]
    CompileError(#[from] CompileError),
    #[error("kernel {kernel:?} is missing from module {module:?}")]
    MissingKernel {
        kernel: Kernel,
        module: KernelModule,
    },
}

/// A rust represenation of a well-defined CUDA module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelModule {
    ElementWiseF32,
}

impl KernelModule {
    #[inline]
    pub const fn source(self) -> &'static str {
        match self {
            Self::ElementWiseF32 => include_str!("../kernels/elementwise_f32.cu"),
        }
    }

    /// A diagnostic filename to pass to runtime compilers.
    #[inline]
    pub const fn source_name(self) -> &'static str {
        match self {
            Self::ElementWiseF32 => "elementwise_f32.cu",
        }
    }
}

/// A rust represenation of a well-defined CUDA kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Kernel {
    AddF32,
}

impl Kernel {
    /// The unmangled function name exported by the CUDA source.
    #[inline]
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::AddF32 => "add_f32",
        }
    }

    #[inline]
    pub const fn module(self) -> KernelModule {
        match self {
            Self::AddF32 => KernelModule::ElementWiseF32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaKernels(HashMap<Kernel, CudaFunction>);

impl CudaKernels {
    #[inline]
    pub fn get(&self, kernel: Kernel) -> Result<&CudaFunction, KernelError> {
        self.0
            .get(&kernel)
            .ok_or_else(|| KernelError::MissingKernel {
                kernel,
                module: kernel.module(),
            })
    }

    pub fn load(
        context: &Arc<CudaContext>,
        kernels: impl IntoIterator<Item = Kernel>,
    ) -> Result<Self, KernelError> {
        let kernels: HashSet<Kernel> = kernels.into_iter().collect();

        let required_modules: HashSet<KernelModule> =
            kernels.iter().map(|kernel| kernel.module()).collect();

        let mut loaded_modules: HashMap<KernelModule, Arc<CudaModule>> =
            HashMap::with_capacity(required_modules.len());

        for module in required_modules {
            let ptx = compile_ptx_with_opts(
                module.source(),
                CompileOptions {
                    name: Some(module.source_name().to_owned()),
                    ..Default::default()
                },
            )?;

            let loaded_module = context.load_module(ptx)?;
            loaded_modules.insert(module, loaded_module);
        }

        let mut loaded_kernels: HashMap<Kernel, CudaFunction> =
            HashMap::with_capacity(kernels.len());

        for kernel in kernels {
            let module = loaded_modules
                .get(&kernel.module())
                .expect("required module was already loaded");

            let function = module.load_function(kernel.entry_point())?;
            loaded_kernels.insert(kernel, function);
        }

        Ok(Self(loaded_kernels))
    }
}

#[cfg(test)]
mod tests {
    use super::Kernel;

    #[test]
    fn add_f32_descriptor_matches_exported_kernel() {
        let kernel = Kernel::AddF32;
        let module = kernel.module();

        assert_eq!(kernel.entry_point(), "add_f32");
        assert_eq!(module.source_name(), "elementwise_f32.cu");
        assert!(
            module
                .source()
                .contains("extern \"C\" __global__ void add_f32")
        );
    }
}

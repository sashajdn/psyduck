use std::{
    num::NonZeroUsize,
    slice,
    time::{Duration, Instant},
};

use cudarc::driver::{CudaStream, sys::CUevent_flags};

pub trait CudaEventSource {
    fn cuda_stream(&self) -> &CudaStream;
}

#[derive(Debug)]
pub struct KernelMetric<T> {
    result: Option<T>,
    elapsed: Duration,
}

impl<T> KernelMetric<T> {
    #[inline]
    pub fn result(&self) -> Option<&T> {
        self.result.as_ref()
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    #[inline]
    pub fn into_result(self) -> Option<T> {
        self.result
    }
}

impl<T> From<KernelMetric<T>> for Vec<KernelMetric<T>> {
    #[inline]
    fn from(metric: KernelMetric<T>) -> Self {
        vec![metric]
    }
}

#[derive(Debug)]
pub struct CudaKernelMetrics<T> {
    metrics: Vec<KernelMetric<T>>,
    total_elapsed: Duration,
    count: usize,
}

impl<T> CudaKernelMetrics<T> {
    pub fn instrument<B, E, GpuOp>(backend: &B, operation: GpuOp) -> Result<Self, E>
    where
        B: CudaEventSource,
        GpuOp: FnOnce(&B) -> Result<T, E>,
        E: From<cudarc::driver::DriverError>,
    {
        let start = Instant::now();
        let metric = Self::sampled_op(backend.cuda_stream(), || operation(backend))?;
        let total_elapsed = start.elapsed();

        Ok(Self {
            metrics: metric.into(),
            total_elapsed,
            count: 1,
        })
    }

    pub fn instrument_many<B, E, GpuOp>(
        backend: &B,
        mut operation: GpuOp,
        count: NonZeroUsize,
        sample_every: Option<NonZeroUsize>,
    ) -> Result<Self, E>
    where
        B: CudaEventSource,
        GpuOp: FnMut(&B) -> Result<T, E>,
        E: From<cudarc::driver::DriverError>,
    {
        let count = count.get();
        let sample_every = sample_every.map_or(1, NonZeroUsize::get);
        let mut metrics = Vec::with_capacity(count.div_ceil(sample_every));
        let stream = backend.cuda_stream();

        let start_time = Instant::now();
        (0..count).try_for_each(|index| -> Result<(), E> {
            if index.is_multiple_of(sample_every) {
                let metric = Self::sampled_op(stream, || operation(backend))?;
                metrics.push(metric);
            } else {
                drop(operation(backend)?);
            }

            Ok(())
        })?;

        stream.synchronize()?;
        let total_elapsed = start_time.elapsed();

        Ok(Self {
            metrics,
            total_elapsed,
            count,
        })
    }

    pub fn instrument_over_many<B, E, GpuOp>(
        backend: &B,
        mut operation: GpuOp,
        count: NonZeroUsize,
    ) -> Result<Self, E>
    where
        B: CudaEventSource,
        GpuOp: FnMut(&B) -> Result<T, E>,
        E: From<cudarc::driver::DriverError>,
    {
        let stream = backend.cuda_stream();
        let start = Instant::now();
        let event_start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

        for _ in 0..count.get() {
            drop(operation(backend)?);
        }

        let event_end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

        let kernel_elapsed = event_start.elapsed_ms(&event_end)?;
        let total_elapsed = start.elapsed();

        Ok(Self {
            metrics: vec![KernelMetric {
                result: None,
                elapsed: Self::cuda_duration(kernel_elapsed),
            }],
            total_elapsed,
            count: count.get(),
        })
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, KernelMetric<T>> {
        self.metrics.iter()
    }

    #[inline]
    pub fn sample_count(&self) -> usize {
        self.metrics.len()
    }

    #[inline]
    pub fn operation_count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn total_elapsed(&self) -> Duration {
        self.total_elapsed
    }

    fn sampled_op<E, GpuOp>(stream: &CudaStream, operation: GpuOp) -> Result<KernelMetric<T>, E>
    where
        GpuOp: FnOnce() -> Result<T, E>,
        E: From<cudarc::driver::DriverError>,
    {
        let event_start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let result = operation()?;
        let event_end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

        Ok(KernelMetric {
            result: Some(result),
            elapsed: Self::cuda_duration(event_start.elapsed_ms(&event_end)?),
        })
    }

    #[inline]
    fn cuda_duration(milliseconds: f32) -> Duration {
        Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0)
    }
}

impl<'a, T> IntoIterator for &'a CudaKernelMetrics<T> {
    type Item = &'a KernelMetric<T>;
    type IntoIter = slice::Iter<'a, KernelMetric<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

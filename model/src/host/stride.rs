use tensor::QuantizedFp;

/// Marks array lengths supported as scalar accumulator lane counts.
pub trait ValidLaneCount {}

impl ValidLaneCount for [(); 1] {}
impl ValidLaneCount for [(); 2] {}
impl ValidLaneCount for [(); 4] {}
impl ValidLaneCount for [(); 8] {}
impl ValidLaneCount for [(); 16] {}
impl ValidLaneCount for [(); 32] {}
impl ValidLaneCount for [(); 64] {}

/// Round-robin scalar accumulators used to expose independent addition chains.
///
/// Only the explicitly supported power-of-two lane counts can be instantiated:
///
/// NOTE: this implementation *did* not properly unroll, eventhough
/// `add_next_lane` is marked `#[inline(always)]`. The compiler was not able to unroll the loop in `sum` because it is a dynamic loop over the lane count.
/// This is why we have the `unroll_four!` macro, which is used in the `sum` method of the `Stride` struct.
pub struct Stride<const LANES: usize, F>
where
    F: QuantizedFp,
    [(); LANES]: ValidLaneCount,
{
    lanes: [F; LANES],
    next: usize,
}

impl<const LANES: usize, F> Stride<LANES, F>
where
    F: QuantizedFp,
    [(); LANES]: ValidLaneCount,
{
    #[inline]
    pub fn zeros() -> Self {
        Self {
            lanes: [F::zero(); LANES],
            next: 0,
        }
    }

    #[inline(always)]
    pub fn add_next_lane(&mut self, value: F) {
        self.lanes[self.next] += value;
        self.next = (self.next + 1) & (LANES - 1);
    }

    #[inline]
    pub fn sum(&self) -> F {
        self.lanes
            .iter()
            .fold(F::zero(), |accumulator, &lane| accumulator + lane)
    }
}

macro_rules! unroll_four {
    ($lane:ident => $body:block) => {{
        {
            const $lane: usize = 0;
            $body
        }
        {
            const $lane: usize = 1;
            $body
        }
        {
            const $lane: usize = 2;
            $body
        }
        {
            const $lane: usize = 3;
            $body
        }
    }};
}
pub(crate) use unroll_four;

#[cfg(test)]
mod tests {
    use super::Stride;

    #[test]
    fn adds_to_lanes_in_round_robin_order() {
        let mut stride = Stride::<4, f32>::zeros();

        for value in 1..=6 {
            stride.add_next_lane(value as f32);
        }

        assert_eq!(stride.lanes, [6.0, 8.0, 3.0, 4.0]);
        assert_eq!(stride.next, 2);
        assert_eq!(stride.sum(), 21.0);
    }
}

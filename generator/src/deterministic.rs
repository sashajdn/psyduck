use tensor::{HostTensor, QuantizedFp, Shape};

use crate::{GeneratorError, Seed};

const A_DOMAIN: Seed = 0x41A4_1A41_5EED_0001;
const B_DOMAIN: Seed = 0xB4B4_1B42_5EED_0002;

pub(crate) const fn a_seed(seed: Seed) -> Seed {
    seed ^ A_DOMAIN
}

pub(crate) const fn b_seed(seed: Seed) -> Seed {
    seed ^ B_DOMAIN
}

pub(crate) fn matrix<F: QuantizedFp>(
    rows: usize,
    columns: usize,
    seed: Seed,
) -> Result<HostTensor<F, 2>, GeneratorError> {
    let element_count = rows
        .checked_mul(columns)
        .ok_or(GeneratorError::DimensionOverflow { rows, columns })?;
    let mut random = SplitMix64::new(seed);
    let values = (0..element_count)
        .map(|_| {
            let signed = random.next_i8();
            F::from_f32(f32::from(signed) / 128.0)
        })
        .collect();

    Ok(HostTensor::from_vec(values, Shape::new([rows, columns]))?)
}

pub(crate) fn exact_matmul_checksum(
    m: usize,
    n: usize,
    k: usize,
    seed: Seed,
) -> Result<f64, GeneratorError> {
    let a_elements = m.checked_mul(k).ok_or(GeneratorError::DimensionOverflow {
        rows: m,
        columns: k,
    })?;
    let b_elements = k.checked_mul(n).ok_or(GeneratorError::DimensionOverflow {
        rows: k,
        columns: n,
    })?;

    let mut a_random = SplitMix64::new(a_seed(seed));
    let mut a_column_sums = vec![0_i64; k];
    for index in 0..a_elements {
        a_column_sums[index % k] += i64::from(a_random.next_i8());
    }

    let mut b_random = SplitMix64::new(b_seed(seed));
    let mut b_row_sums = vec![0_i64; k];
    for index in 0..b_elements {
        b_row_sums[index / n] += i64::from(b_random.next_i8());
    }

    let numerator = a_column_sums
        .into_iter()
        .zip(b_row_sums)
        .map(|(a, b)| i128::from(a) * i128::from(b))
        .sum::<i128>();

    Ok(numerator as f64 / 16_384.0)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_i8(&mut self) -> i8 {
        (self.next() >> 56) as u8 as i8
    }
}

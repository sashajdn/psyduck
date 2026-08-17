mod deterministic;

use tensor::{HostTensor, QuantizedFp};

pub type Seed = u64;

pub const CANONICAL_SEED: Seed = 0x5053_5944_5543_4B01;
pub const CANONICAL_MATMUL_CHECKSUMS: [(usize, f64); 11] = [
    (4, 5.818_115_234_375),
    (8, 9.721_069_335_937_5),
    (16, -2.028_442_382_812_5),
    (32, 11.211_242_675_781_25),
    (64, -69.861_328_125),
    (128, 350.061_401_367_187_5),
    (256, -342.720_458_984_375),
    (512, -1_873.092_895_507_812_5),
    (1_024, 1_308.468_261_718_75),
    (2_048, 129_273.643_493_652_34),
    (4_096, 939_875.690_734_863_3),
];

pub fn canonical_matmul_checksum(size: usize) -> Option<f64> {
    CANONICAL_MATMUL_CHECKSUMS
        .iter()
        .find_map(|(candidate, checksum)| (*candidate == size).then_some(*checksum))
}

pub fn calculate_matmul_checksum(
    m: usize,
    n: usize,
    k: usize,
    seed: Seed,
) -> Result<f64, GeneratorError> {
    deterministic::exact_matmul_checksum(m, n, k, seed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Generator {
    Deterministic(Option<Seed>),
    NonDeterministic,
}

pub struct GeneratedMatrices<F: QuantizedFp> {
    pub a: HostTensor<F, 2>,
    pub b: HostTensor<F, 2>,
    pub seed: Seed,
}

impl Default for Generator {
    fn default() -> Self {
        Self::Deterministic(None)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeneratorError {
    #[error("matrix dimensions overflow: {rows} x {columns}")]
    DimensionOverflow { rows: usize, columns: usize },
    #[error(transparent)]
    ElementCountMismatch(#[from] tensor::ElementCountMismatch),
}

impl Generator {
    #[must_use]
    pub const fn is_canonical(self) -> bool {
        matches!(self, Self::Deterministic(None | Some(CANONICAL_SEED)))
    }

    /// Builds A[M,K] and B[K,N].
    pub fn matrices<F: QuantizedFp>(
        &self,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<GeneratedMatrices<F>, GeneratorError> {
        let seed = match self {
            Self::Deterministic(seed) => seed.unwrap_or(CANONICAL_SEED),
            Self::NonDeterministic => rand::random(),
        };

        Ok(GeneratedMatrices {
            a: deterministic::matrix(m, k, deterministic::a_seed(seed))?,
            b: deterministic::matrix(k, n, deterministic::b_seed(seed))?,
            seed,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Deterministic(_) => "deterministic",
            Self::NonDeterministic => "non-deterministic",
        }
    }
}

#[cfg(test)]
mod tests {
    use tensor::Tensor;

    use super::{CANONICAL_MATMUL_CHECKSUMS, CANONICAL_SEED, Generator, calculate_matmul_checksum};

    #[test]
    fn canonical_seed_is_used_by_default() {
        let default = Generator::default().matrices::<f32>(2, 3, 4).unwrap();
        let explicit = Generator::Deterministic(Some(CANONICAL_SEED))
            .matrices::<f32>(2, 3, 4)
            .unwrap();

        assert_eq!(default.seed, CANONICAL_SEED);
        assert_eq!(default.a.as_slice(), explicit.a.as_slice());
        assert_eq!(default.b.as_slice(), explicit.b.as_slice());
        assert_eq!(
            &default.a.as_slice()[..6],
            &[-0.8125, -0.34375, -0.2734375, 0.28125, -0.21875, -0.390625]
        );
    }

    #[test]
    fn builds_m_by_k_and_k_by_n_matrices() {
        let matrices = Generator::default().matrices::<f64>(2, 3, 4).unwrap();

        assert_eq!(matrices.a.shape().dims(), &[2, 4]);
        assert_eq!(matrices.b.shape().dims(), &[4, 3]);
    }

    #[test]
    fn canonical_checksums_match_the_exact_integer_workload() {
        for (size, expected) in CANONICAL_MATMUL_CHECKSUMS {
            let actual = calculate_matmul_checksum(size, size, size, CANONICAL_SEED).unwrap();
            assert_eq!(actual, expected, "canonical checksum changed for N={size}");
        }
    }
}

use generator::{CANONICAL_SEED, calculate_matmul_checksum};

const SIZES: [usize; 11] = [4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096];

fn main() {
    for size in SIZES {
        let checksum = calculate_matmul_checksum(size, size, size, CANONICAL_SEED)
            .expect("canonical checksum should be calculable");
        println!("({size}, {checksum:?}),");
    }
}

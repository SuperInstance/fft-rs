# fft-rs

Fast Fourier Transform implementations in pure Rust.

## Features

- **Cooley-Tukey radix-2 FFT** (recursive)
- **Iterative FFT** (in-place, bit-reversal)
- **Inverse FFT** (IFFT)
- **Circular and linear convolution** via FFT
- **Discrete Cosine Transform** (DCT-II and inverse)

## Usage

```rust
use fft_rs::{fft, inverse_fft, Complex};

let input = vec![
    Complex::new(1.0, 0.0),
    Complex::new(2.0, 0.0),
    Complex::new(3.0, 0.0),
    Complex::new(4.0, 0.0),
];

let spectrum = fft(&input);
let recovered = inverse_fft(&spectrum);
```

## License

MIT OR Apache-2.0

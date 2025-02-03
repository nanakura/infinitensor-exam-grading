use std::iter::Sum;

use cudarc::cublas::Gemm;
use cudarc::cublas::{CudaBlas, GemmConfig};
use cudarc::driver::CudaView;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::driver::{CudaFunction, CudaSlice};
use half::{bf16, f16};
use num_traits::Float;
use std::sync::Arc;

use crate::tensor::Tensor;

mod cuda_kernels;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

pub trait CudaDType: Sized + cudarc::driver::DeviceRepr {
    const DTYPE: DType;
    fn gemm(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        a: &CudaView<Self>,
        b: &CudaView<Self>,
        c: &mut CudaSlice<Self>,
    );
}

impl CudaDType for f32 {
    const DTYPE: DType = DType::F32;
    fn gemm(
        blas: &CudaBlas,
        cfg: GemmConfig<f32>,
        a: &CudaView<f32>,
        b: &CudaView<f32>,
        c: &mut CudaSlice<f32>,
    ) {
        unsafe {
            blas.gemm(cfg, a, b, c).unwrap();
        }
    }
}

impl CudaDType for f16 {
    const DTYPE: DType = DType::F16;
    fn gemm(
        blas: &CudaBlas,
        cfg: GemmConfig<f16>,
        a: &CudaView<f16>,
        b: &CudaView<f16>,
        c: &mut CudaSlice<f16>,
    ) {
        unsafe {
            blas.gemm(cfg, a, b, c).unwrap();
        }
    }
}

impl CudaDType for bf16 {
    const DTYPE: DType = DType::BF16;
    fn gemm(
        blas: &CudaBlas,
        cfg: GemmConfig<bf16>,
        a: &CudaView<bf16>,
        b: &CudaView<bf16>,
        c: &mut CudaSlice<bf16>,
    ) {
        unsafe {
            blas.gemm(cfg, a, b, c).unwrap();
        }
    }
}

#[derive(Clone)]
pub struct CudaOperator {
    dev: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

fn kernel_name<T: CudaDType>(base_name: &str) -> String {
    let dtype_str = match T::DTYPE {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
    };
    format!("{base_name}_{dtype_str}")
}

impl CudaOperator {
    pub fn new() -> Self {
        let dev = CudaDevice::new(0).unwrap();
        let blas = Arc::new(CudaBlas::new(dev.clone()).unwrap());

        Self { dev, blas }
    }

    fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> CudaFunction {
        if !self.dev.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.dev
                .load_ptx(ptx.into(), module_name, &[static_module_name])
                .unwrap();
        }
        let func = self.dev.get_func(module_name, module_name).unwrap();
        // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
        // able to only build the error value if needed.
        func
    }
    // pub fn cos<T: Copy + Default + Float + OpDType>(&self, y: &mut Tensor<T>) {
    //     let kname = kernel_name::<T>("ucos");
    //     let f = self.get_or_load_func(&kname, candle_kernels::UNARY);
    //     let len = y.data().len();
    //     let cfg = LaunchConfig::for_num_elems(len as u32);
    //     let y_data = unsafe{y.data_mut()};
    //     let mut y_dev = self.dev.htod_sync_copy(y_data).unwrap();
    //     let params = (len, 1usize, 0usize, 0usize, &mut y_dev);
    //     unsafe { f.launch(cfg, params).unwrap() };
    //     self.dev.dtoh_sync_copy_into(&y_dev, y_data).unwrap();
    // }

    // pub fn sin<T: Copy + Default + Float + OpDType>(&self, y: &mut Tensor<T>) {
    //     let kname = kernel_name::<T>("usin");
    //     let f = self.get_or_load_func(&kname, candle_kernels::UNARY);
    //     let len = y.data().len();
    //     let cfg = LaunchConfig::for_num_elems(len as u32);
    //     let y_data = unsafe{y.data_mut()};
    //     let mut y_dev = self.dev.htod_sync_copy(y_data).unwrap();
    //     let params = (len, 1usize, 0usize, 0usize, &mut y_dev);
    //     unsafe { f.launch(cfg, params).unwrap() };
    //     self.dev.dtoh_sync_copy_into(&y_dev, y_data).unwrap();
    // }

    // C = beta * C + alpha * A @ B^T
    pub fn matmul_transb<T: CudaDType + Copy + Default>(
        &self,
        c: &mut Tensor<T>,
        beta: T,
        a: &Tensor<T>,
        b: &Tensor<T>,
        alpha: T,
    ) {
        let (m, n) = (a.shape()[0], a.shape()[1]);
        let p = b.shape()[0];
        let kname = kernel_name::<T>("matmul_transb");
        let f = self.get_or_load_func(&kname, cuda_kernels::MATMUL_TRANSB);

        // Convert tensors to f32 slices
        let a_host: &[T] =
            unsafe { std::slice::from_raw_parts(a.data().as_ptr() as *const T, a.data().len()) };
        let b_host: &[T] =
            unsafe { std::slice::from_raw_parts(b.data().as_ptr() as *const T, b.data().len()) };
        let c_host: &mut [T] = unsafe {
            std::slice::from_raw_parts_mut(c.data_mut().as_mut_ptr() as *mut T, c.data().len())
        };

        // Copy data to device
        let a_dev = self.dev.htod_sync_copy(a_host).unwrap();
        let b_dev = self.dev.htod_sync_copy(b_host).unwrap();
        let mut c_dev = self.dev.htod_sync_copy(c_host).unwrap();

        // Configure kernel launch parameters
        let block_dim = (16, 16, 1);
        let grid_dim = (
            (p as u32 + block_dim.0 - 1) / block_dim.0,
            (m as u32 + block_dim.1 - 1) / block_dim.1,
            1,
        );

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            f.launch(
                cfg,
                (
                    &mut c_dev, beta, &a_dev, &b_dev, alpha, m as i32, n as i32, p as i32,
                ),
            )
            .unwrap();
        }

        // Copy result back to host
        self.dev.dtoh_sync_copy_into(&c_dev, c_host).unwrap();
        self.dev.synchronize().unwrap();
    }

    pub fn rms_norm<T: CudaDType + Copy + Default + Float>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<T>,
        w: &Tensor<T>,
        epsilon: T,
    ) {
        //assert_eq!(y.shape(), x.shape());
        let x_shape = x.shape();
        let (m, n) = match x_shape.len() {
            1 => (1, x_shape[0]),          // 处理向量输入
            2 => (x_shape[0], x_shape[1]), // 处理矩阵输入
            _ => panic!("Unsupported tensor shape"),
        };

        let x_host = x.data();
        let w_host = w.data();
        let y_host: &mut [T] = unsafe { y.data_mut() };
        let x_dev = self.dev.htod_sync_copy(x_host).unwrap();
        let w_dev = self.dev.htod_sync_copy(w_host).unwrap();
        let mut y_dev = self.dev.htod_sync_copy(y_host).unwrap();

        let kname = kernel_name::<T>("rms_norm");
        let func = self.get_or_load_func(&kname, cuda_kernels::RMSNORM);

        let block_dim = 256;
        let grid_dim = m as u32;

        let shared_mem_bytes = block_dim * std::mem::size_of::<T>() as u32;

        let cfg = LaunchConfig {
            block_dim: (block_dim, 1, 1),
            grid_dim: (grid_dim, 1, 1),
            shared_mem_bytes,
        };

        let params = (&mut y_dev, &x_dev, &w_dev, epsilon, m as i32, n as i32);
        unsafe { func.launch(cfg, params).unwrap() };

        self.dev.dtoh_sync_copy_into(&y_dev, y_host).unwrap();
    }

    // y = silu(x) * y
    // hint: this is an element-wise operation
    pub fn swiglu<T: Copy + Default + Float + Sum + CudaDType>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<T>,
    ) {
        assert_eq!(y.size(), x.size());
        let num_elements = y.size();
        let kname = kernel_name::<T>("swiglu");
        let f = self.get_or_load_func(&kname, cuda_kernels::SWIGLU);

        let x_host: &[T] = x.data();
        let y_host: &mut [T] = unsafe { y.data_mut() };

        let x_dev = self.dev.htod_sync_copy(x_host).unwrap();
        let mut y_dev = self.dev.htod_sync_copy(y_host).unwrap();

        let block_dim = 256;
        let grid_dim = (num_elements as u32 + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            block_dim: (block_dim, 1, 1),
            grid_dim: (grid_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(cfg, (&mut y_dev, &x_dev, num_elements as i32))
                .unwrap();
        }

        self.dev.dtoh_sync_copy_into(&y_dev, y_host).unwrap();
    }

    pub fn rope<T: Copy + Default + Float + CudaDType>(
        &self,
        y: &mut Tensor<T>,
        start_pos: usize,
        theta: T,
    ) {
        let shape = y.shape();
        assert!(shape.len() == 3);
        let seq_len = shape[0];
        let n_heads = shape[1];
        let d = shape[2];

        let kname = kernel_name::<T>("rope");
        let f = self.get_or_load_func(&kname, cuda_kernels::ROPE);

        let data_host: &mut [T] = unsafe {
            std::slice::from_raw_parts_mut(y.data_mut().as_mut_ptr() as *mut T, y.data().len())
        };

        let mut data_dev = self.dev.htod_sync_copy(data_host).unwrap();

        let block_dim = (8, 8, 8);
        let grid_dim = (
            (seq_len as u32 + block_dim.0 - 1) / block_dim.0,
            (n_heads as u32 + block_dim.1 - 1) / block_dim.1,
            (d as u32 / 2 + block_dim.2 - 1) / block_dim.2,
        );

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(
                cfg,
                (
                    &mut data_dev,
                    start_pos as i32,
                    theta.to_f32().unwrap(),
                    seq_len as i32,
                    n_heads as i32,
                    d as i32,
                ),
            )
            .unwrap();
        }

        self.dev.dtoh_sync_copy_into(&data_dev, data_host).unwrap();
    }

    pub fn gather<T: Copy + Default + CudaDType>(
        &self,
        y: &mut Tensor<T>,
        indices: &Tensor<u32>,
        table: &Tensor<T>,
    ) {
        let length = indices.size();
        let table_shape = table.shape();
        assert!(table_shape.len() == 2);
        let dim = table_shape[1];
        assert!(y.size() == length * dim);

        let kname = kernel_name::<T>("gather");
        let f = self.get_or_load_func(&kname, cuda_kernels::GATHER);

        let indices_host: &[u32] = indices.data();
        let table_host = unsafe {
            std::slice::from_raw_parts(table.data().as_ptr() as *const T, table.data().len())
        };
        let y_host = unsafe {
            std::slice::from_raw_parts_mut(y.data_mut().as_mut_ptr() as *mut T, y.data().len())
        };

        let indices_dev = self.dev.htod_sync_copy(indices_host).unwrap();
        let table_dev = self.dev.htod_sync_copy(table_host).unwrap();
        let mut y_dev = self.dev.htod_sync_copy(y_host).unwrap();

        let block_dim = (16, 16, 1);
        let grid_dim = (
            (length as u32 + block_dim.0 - 1) / block_dim.0,
            (dim as u32 + block_dim.1 - 1) / block_dim.1,
            1,
        );

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(
                cfg,
                (
                    &mut y_dev,
                    &indices_dev,
                    &table_dev,
                    length as i32,
                    dim as i32,
                ),
            )
            .unwrap();
        }

        self.dev.dtoh_sync_copy_into(&y_dev, y_host).unwrap();
    }

    // softmax(x) = exp(x - max) / sum(exp(x - max))
    // y = softmax(mask(x))
    pub fn masked_softmax<T: Copy + Default + Float + Sum + CudaDType>(&self, y: &mut Tensor<T>) {
        let ndim = y.shape().len();
        assert!(ndim >= 2);
        let seq_len = y.shape()[ndim - 2];
        let total_seq_len = y.shape()[ndim - 1];
        let batch = y.size() / (seq_len * total_seq_len);
        let kname = kernel_name::<T>("masked_softmax");
        let f = self.get_or_load_func(&kname, cuda_kernels::SOFTMAX);

        let data_host = unsafe {
            std::slice::from_raw_parts_mut(y.data_mut().as_mut_ptr() as *mut T, y.data().len())
        };
        let mut data_dev = self.dev.htod_sync_copy(data_host).unwrap();

        // Configure kernel launch parameters
        let block_dim = (1, 256, 1);
        let grid_dim = (
            batch as u32,
            (seq_len as u32 + block_dim.1 - 1) / block_dim.1,
            1,
        );

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(
                cfg,
                (
                    &mut data_dev,
                    batch as i32,
                    seq_len as i32,
                    total_seq_len as i32,
                ),
            )
            .unwrap();
        }

        self.dev.dtoh_sync_copy_into(&data_dev, data_host).unwrap();
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax<T: Copy + Default + Float + Sum>(y: &mut Tensor<T>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<T>();

            (0..boundary).for_each(|j| data[offset + j] = data[offset + j] / sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = T::zero());
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot<T: Copy + Default + Float>(x: &Tensor<T>, y: &Tensor<T>) -> T {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = T::zero();
    for i in 0..len {
        sum = sum + x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample<T: Copy + Default + Float>(
    x: &Tensor<T>,
    top_p: T,
    top_k: u32,
    temperature: T,
) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= T::zero() || top_k < 2 || top_p <= T::zero() {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl<T: Float> From<(usize, &T)> for Probability {
        #[inline]
        fn from((i, p): (usize, &T)) -> Self {
            Self {
                val: p.to_f32().unwrap(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val =
            logits[i - 1].val + ((logits[i].val - max) / temperature.to_f32().unwrap()).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p.to_f32().unwrap();
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    let operator = CudaOperator::new();
    operator.swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    let operator = CudaOperator::new();
    operator.rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);

    let operator = CudaOperator::new();
    operator.matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_mulmat_transb_comprehensive() {
    // Test 1: 1x1 matrices
    let mut c = Tensor::<f32>::new(vec![2.], &vec![1, 1]);
    let a = Tensor::<f32>::new(vec![3.], &vec![1, 1]);
    let b = Tensor::<f32>::new(vec![4.], &vec![1, 1]);
    let operator = CudaOperator::new();
    operator.matmul_transb(&mut c, 0.5, &a, &b, 2.0);
    assert!(c.close_to(&Tensor::<f32>::new(vec![25.], &vec![1, 1]), 1e-3));

    // Test 2: Zero matrix
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![0., 0., 0., 0., 0., 0.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    operator.matmul_transb(&mut c, 1.0, &a, &b, 1.0);
    assert!(c.close_to(&Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]), 1e-3));

    // Test 3: Identity matrix
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 0., 0., 1.], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![1., 0., 0., 1.], &vec![2, 2]);
    operator.matmul_transb(&mut c, 0.0, &a, &b, 1.0);
    assert!(c.close_to(&Tensor::<f32>::new(vec![1., 0., 0., 1.], &vec![2, 2]), 1e-3));

    // Test 4: Large matrices and precision
    let mut c = Tensor::<f32>::new(vec![0.; 16], &vec![4, 4]);
    let a = Tensor::<f32>::new((0..32).map(|x| x as f32).collect(), &vec![4, 8]);
    let b = Tensor::<f32>::new((0..32).map(|x| x as f32).collect(), &vec![4, 8]);
    operator.matmul_transb(&mut c, 0.0, &a, &b, 1.0);
    let expected = Tensor::<f32>::new(
        vec![
            140., 364., 588., 812., 364., 1100., 1836., 2572., 588., 1836., 3084., 4332., 812.,
            2572., 4332., 6092.,
        ],
        &vec![4, 4],
    );
    assert!(c.close_to(&expected, 1e-3));

    // Test 5: Different alpha and beta combinations
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    operator.matmul_transb(&mut c, 2.0, &a, &b, 3.0);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![17., 37., 39., 83.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
pub fn test_mulmat_transb2() {
    let a = Tensor::<f32>::new(
        vec![
            0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995,
        ],
        &vec![4, 2],
    );
    let b = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![3, 2]);
    let mut c = Tensor::<f32>::default(&vec![4, 3]);

    let op = CudaOperator::new();
    op.matmul_transb(&mut c, 0., &a, &b, 1.);
    //println!("{:?}", c.data());
    assert!(c.close_to(
        &Tensor::<f32>::new(
            vec![
                0.29999986, 0.6999997, 1.0999994, 0.29999986, 0.6999997, 1.0999994, 0.29999986,
                0.6999997, 1.0999994, 0.29999986, 0.6999997, 1.0999994
            ],
            &vec![4, 3]
        ),
        1e-3
    ));
}

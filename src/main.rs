//!
#![warn(missing_debug_implementations, rust_2018_idioms, missing_docs)]

use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;
use std::fs;
//#[macro_use]
//extern crate ndarray;

//#[macro_use]
//extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::*;

#[derive(Serialize, Deserialize)]
struct PersonPose {
    id: usize,
    keypoints3d: [[f64; 4]; 25],
}

impl fmt::Debug for PersonPose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersonPose")
            .field("id", &self.id)
            .field("keypoints3d", &self.keypoints3d)
            .finish()
    }
}

fn load_poses_from_json(data: &str) -> Option<Vec<PersonPose>> {
    let v: Vec<PersonPose> = serde_json::from_str(data).ok()?;
    // Access parts of the data by indexing with square brackets.
    std::option::Option::Some(v)
}

fn save_poses_to_json(p: &[PersonPose]) -> String {
    // Access parts of the data by indexing with square brackets.
    let s: String = serde_json::to_string_pretty(&p).ok().unwrap();
    s
}

fn personpose_to_mat(p: &PersonPose, mask: &[bool]) -> Array<f64, Dim<[usize; 2]>> {
    let msize: usize = mask
        .iter()
        .fold(0, |msize, x| msize + (if *x { 1 } else { 0 }));
    let mut a = Array::zeros((msize, 3));

    let mut ind = 0;
    for (i, mask_i) in mask.iter().enumerate() {
        if *mask_i {
            for j in 0..3 {
                a[(ind, j)] = p.keypoints3d[i][j]
            }
            ind += 1;
        }
    }
    a
}

fn mat_to_personpose(p_id: usize, mat: &Array<f64, Dim<[usize; 2]>>) -> PersonPose {
    let mut arr: [[f64; 4]; 25] = [[0.0; 4]; 25];
    for i in 0..25 {
        for j in 0..4 {
            arr[i][j] = mat[(i, j)];
        }
    }
    PersonPose {
        id: p_id,
        keypoints3d: arr,
    }
}

fn complete_with_transformed_ref(
    p: &PersonPose,
    ref_p: &PersonPose,
) -> Array<f64, Dim<[usize; 2]>> {
    let mask: Vec<bool> = (0..25).map(|x| p.keypoints3d[x][3] >= 0.8).collect();
    let ref_mat = personpose_to_mat(ref_p, &mask);
    let mu_ref = ref_mat.mean_axis(Axis(0));
    let ref_mat = ref_mat - mu_ref.as_ref().unwrap();
    let a_mat = personpose_to_mat(p, &mask);

    //find rotation, translation
    let (rotation, translation) = procrustes(&a_mat, &ref_mat);

    let rotation = rotation_matrix_z_component(&rotation);
    let confidence = 0.7;
    let mut completed_keypoints = Array::zeros((25, 4));
    let mask: Vec<bool> = (0..25).map(|_x| true).collect();
    let complete_ref_mat = personpose_to_mat(ref_p, &mask);
    let _mu_complete_ref = complete_ref_mat.mean_axis(Axis(0));
    let complete_ref_mat = complete_ref_mat - mu_ref.as_ref().unwrap();

    for i in 0..25 {
        //if p.keypoints3d[i][3] == 0.0{
        if p.keypoints3d[i][3] < 0.7 {
            let mut transformed_point = Array::zeros((1, 3));
            for j in 0..3 {
                transformed_point[(0, j)] = complete_ref_mat[(i, j)]; //ref_p.keypoints3d[i][j];
            }
            //println!(": {:?}", check);
            //println!("det: {:?}", rotation.det());
            // * rotation) + translation;
            //transformed_point = transformed_point * rotation + translation;
            //transformed_point = rotation * transformed_point;
            let res = (transformed_point.dot(&rotation)) + &translation;
            //println!("to: {:?}", res);

            //let res = ((-trans + (rot*p)) - b.slice(s![0, ..]));
            for j in 0..3 {
                completed_keypoints[(i, j)] = res[(0, j)];
                completed_keypoints[(i, 3)] = confidence;
            }
        } else {
            for j in 0..4 {
                completed_keypoints[(i, j)] = p.keypoints3d[i][j];
            }
        }
    }
    completed_keypoints
}

//https://tipsfordev.com/procrustes-analysis-with-numpy
fn procrustes(x_in: &Array2<f64>, y_in: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    assert_eq!(x_in.shape(), y_in.shape());

    let mu_x = x_in.mean_axis(Axis(0));
    let mu_y = y_in.mean_axis(Axis(0));
    let x_0 = x_in - mu_x.as_ref().unwrap();
    let y_0 = y_in - mu_y.as_ref().unwrap();
    let ss_x = x_0.mapv(|a| a.powi(2)).sum();
    let ss_y = y_0.mapv(|a| a.powi(2)).sum();

    // centered froebius norm
    let norm_x = ss_x.sqrt();
    let norm_y = ss_y.sqrt();

    // scale to equal (unit) norm
    let x_0 = x_0 / norm_x;
    let y_0 = y_0 / norm_y;

    let axy = x_0.t().dot(&y_0);
    //U, s, Vt =
    let (u, _s, vt) = axy.svd(true, true).unwrap(); //ndarray_linalg::svd::SVD{a};
    let vt = vt.unwrap();
    let v = vt.t();

    //let trace_ta = s.sum();

    let b = 1.0;
    // these are needed for scalling
    //let d = 1.0 + ss_y/ss_x -2.0 * trace_ta * norm_y / norm_x;
    //let z = norm_y * y_0.dot(&t) + mu_x.as_ref().unwrap();

    let rotation = v.dot(&u.unwrap().t());
    let translation = mu_x.unwrap() - b * mu_y.unwrap().dot(&rotation);

    (rotation, translation)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];
    //println!("{:?}", args);

    //    let refPersonPose = PersonPose{id:0, keypoints3d:[[1.0,0.0,0.0,0.0]]};
    //println!("In file {}", filename);

    let ref_filename = "complete_pose_reference.json";
    let ref_contents =
        fs::read_to_string(ref_filename).expect("Something went wrong reading the file");
    let ref_pose = load_poses_from_json(&ref_contents).unwrap();

    let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");
    let a = load_poses_from_json(&contents).unwrap();

    //create array for procrustes
    let mut v: Vec<PersonPose> = Vec::new();
    for (count, ps) in a.into_iter().enumerate() {
        //println!("orig: {:?}", &a_mat);
        //println!("ref: {:?}", &ref_mat);
        //println!("rot: {:?}", &rotation);
        //println!("trans: {:?}", &translation);
        //apply rot/transl to ref and complete a
        let completed = complete_with_transformed_ref(&ps, &ref_pose[0]);
        //println!("{:?}", completed);

        let p = mat_to_personpose(count, &completed);
        v.push(p);
        //println!("{}",count);
    }
    //let v: Vec<PersonPose> = vec![p];
    let s = save_poses_to_json(&v);
    let out_filename = "out/".to_owned() + filename;
    fs::write(out_filename, &s).expect("Unable to write file");
}

//https://learnopencv.com/rotation-matrix-to-euler-angles/
// Checks if a matrix is a valid rotation matrix.
#[cfg(test)]
fn is_rotation_matrix(r: &Array<f64, Dim<[usize; 2]>>) -> bool {
    println!("{:?}", r);
    let i_mat: Array<f64, Dim<[usize; 2]>> = Array2::eye(3);
    let check = i_mat.dot(r).dot(&r.t());
    let i_mat: Array<f64, Dim<[usize; 2]>> = Array2::eye(3);
    (check - i_mat).norm() < 0.000001
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
fn rotation_matrix_z_component(r: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    //float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0)
    // +  R.at<double>(1,0) * R.at<double>(1,0) );

    let sy = (r[(0, 0)].powi(2) + r[(1, 0)].powi(2)).sqrt();

    let singular = sy < 1.0e-6_f64;
    //let mut x: f64;
    //let mut y: f64;
    let z: f64;
    if !singular {
        // x = r[(2, 1)].atan2(r[(2, 2)]);
        // y = (-r[(2, 0)]).atan2(sy);
        z = r[(1, 0)].atan2(r[(0, 0)]);
    } else {
        // x = (-r[(1, 2)]).atan2(r[(1, 1)]);
        // y = (-r[(2, 0)]).atan2(sy);
        z = 0.0;
    }

    // Calculate rotation about z axis
    //Mat R_z = (Mat_<double>(3,3) <<
    //           cos(theta[2]),    -sin(theta[2]),      0,
    //           sin(theta[2]),    cos(theta[2]),       0,
    //           0,               0,                  1);
    array![
        [z.cos(), -z.sin(), 0.0],
        [z.sin(), z.cos(), 0.0],
        [0.0, 0.0, 1.0]
    ]
}

#[test]
fn procrustes_works() {
    let a = array![[1.0, 0.0, 2.0], [0.0, 0.0, -2.0]];
    let b = array![[2.0, 0.0, 1.0], [-2.0, 0.0, 0.0]];
    let correct_r = array![
        [-5.34384992e-17, 0.00000000e+00, 1.00000000e+00],
        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
        [1.00000000e+00, 0.00000000e+00, -7.85941422e-17]
    ];

    for i in 0..2 {
        let (rot, trans) = procrustes(&a, &b);
        let p = a.slice(s![i, ..]);
        let res: Array<f64, Dim<[usize; 1]>> = (-trans + (rot.dot(&p))) - b.slice(s![i, ..]);
        assert!(res.norm() < 0.000001);
    }
    let (rot, _trans) = procrustes(&a, &b);
    //assert_eq!((rot - correct_r).sum() < 0.001, true);
    assert!(is_rotation_matrix(&rot));
    assert!((rot - correct_r).norm() < 0.00001);
}

#[test]
fn load_json() {
    let ref_filename = "complete_pose_reference.json";
    let ref_contents =
        fs::read_to_string(ref_filename).expect("Something went wrong reading the file");
    let ref_pose = load_poses_from_json(&ref_contents).unwrap();
    assert!(ref_pose[0].keypoints3d[0][0] != 0.0);
}

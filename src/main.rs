//!
#![warn(missing_debug_implementations, rust_2018_idioms, missing_docs)]

use std::env;
use std::fmt;
use std::fs;
use serde::{Deserialize, Serialize};
//#[macro_use]
//extern crate ndarray;

//#[macro_use]
//extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray::OwnedRepr;

#[derive(Serialize, Deserialize)]
struct PersonPose {
    id: u32,
    keypoints3d: [[f32; 4]; 25],
    //Vec<Vec<f32>>,
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

fn save_poses_to_json(p: &Vec<PersonPose> ) -> String {
    // Access parts of the data by indexing with square brackets.
    let s: String = serde_json::to_string_pretty(&p).ok().unwrap();
    s
}



fn personpose_to_mat(p: &PersonPose, mask: &Vec<bool>) -> ndarray::ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>{
    let msize: usize =  mask.iter().fold(0, |msize, x| msize + (if *x {1} else {0}));
    let mut a = Array::zeros((msize, 3));
    let mut ind = 0;
    for i in 0..msize{
        if mask[i] {
            for j in 0..3{
                a[(ind, j)] = p.keypoints3d[i][j]
            }
            ind += 1;
        }
    }
    a
}


fn mat_to_personpose(p_id: u32, mat: &ndarray::ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>) -> PersonPose{
    let mut arr: [[f32; 4]; 25] = [[0.0; 4]; 25];
    for i in 0..25 {
        for j in 0..4 {
            arr[i][j] = mat[(i,j)];
        }
    }
    let p = PersonPose{id: p_id, keypoints3d: arr};
    p
}

fn complete_with_transformed_ref(p: &PersonPose, ref_p: &PersonPose, transformation: (&ndarray::ArrayBase<OwnedRepr<f32>,
                                                                     ndarray::Dim<[usize; 2]>>,
                                                  &ndarray::ArrayBase<OwnedRepr<f32>,
                                                                     ndarray::Dim<[usize; 1]>>)
) -> ndarray::ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>{
    let (rotation, translation) = transformation;
    let confidence = 0.5;
    let mut completed_keypoints = Array::zeros((25, 4));
    for i in 0..25 {
        if p.keypoints3d[i][3] == 0.0{
            let mut transformed_point = Array::zeros((3,1));
            for j in 0..3{
                transformed_point[(j,0)] = ref_p.keypoints3d[i][j];
            }
            // * rotation) + translation;
            transformed_point = transformed_point * rotation + translation;
            for j in 0..3{
                completed_keypoints[(i,j)] = transformed_point[(0,j)];
                completed_keypoints[(i,3)] = confidence;
            }
        } else {
            for j in 0..4{
                completed_keypoints[(i,j)]=p.keypoints3d[i][j];
            }
        }
    }
    completed_keypoints
}


//https://tipsfordev.com/procrustes-analysis-with-numpy
fn procrustes(x: &Array2<f32>, y: &Array2<f32>) -> (ndarray::ArrayBase<OwnedRepr<f32>,
                                                                     ndarray::Dim<[usize; 2]>>,
                                                  ndarray::ArrayBase<OwnedRepr<f32>,
                                                                     ndarray::Dim<[usize; 1]>>)
{
    assert_eq!(x.shape(), y.shape());

    let mu_x = x.mean_axis(Axis(0));
    let mu_y = y.mean_axis(Axis(0));
    let x_0 = x - mu_x.as_ref().unwrap();
    let y_0 = y - mu_y.as_ref().unwrap();
    let ss_x = x_0.mapv(|a| a.powi(2)).sum();
    let ss_y = y_0.mapv(|a| a.powi(2)).sum();

    // centered froebius norm
    let norm_x = ss_x.sqrt();
    let norm_y = ss_y.sqrt();

    // scale to equal (unit) norm
    let x_0 = x_0 / norm_x;
    let y_0 = y_0 / norm_y;

    let a = x_0.t().dot(&y_0);
    //U, s, Vt =
    let (u, _s, vt) = a.svd(true, true).unwrap();//ndarray_linalg::svd::SVD{a};
    let vt = vt.unwrap();
    let v = vt.t();
    let t = v.dot(&u.unwrap().t());

    //let trace_ta = s.sum();

    let b = 1.0;
    // these are needed for scalling
    //let d = 1.0 + ss_y/ss_x -2.0 * trace_ta * norm_y / norm_x;
    //let z = norm_y * y_0.dot(&t) + mu_x.as_ref().unwrap();

    let c = mu_x.unwrap() - b* mu_y.unwrap().dot(&t);

    let rotation = t;
    let translation = c;
    return (rotation, translation);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];
    //println!("{:?}", args);

//    let refPersonPose = PersonPose{id:0, keypoints3d:[[1.0,0.0,0.0,0.0]]};
    //println!("In file {}", filename);

    let contents = fs::read_to_string(filename)
        .expect("Something went wrong reading the file");

    let ref_filename = "complete_pose_reference.json";
    let ref_contents = fs::read_to_string(ref_filename)
        .expect("Something went wrong reading the file");
    let ref_pose = load_poses_from_json(&ref_contents).unwrap();

    //let _a = untyped_example(&contents);
    let a = load_poses_from_json(&contents).unwrap();

    //create array for procrustes
    let mask: Vec<bool> = (0..25).map(|x| a[0].keypoints3d[x][3]!=0.0).collect();
    let ref_mat = personpose_to_mat(&ref_pose[0], &mask);
    let a_mat = personpose_to_mat(&a[0], &mask);

    //find rotation, translation
    let (rotation, translation) = procrustes(&a_mat, &ref_mat);
    //println!("rotation {:?}", rotation);
    //println!("translation {:?}", translation);

    //apply rot/transl to ref and complete a
    //println!("{:?}", a_mat);
    //println!("{:?}", ref_mat);
    let completed = complete_with_transformed_ref(&a[0], &ref_pose[0], (&rotation, &translation));
    //println!("{:?}", completed);

    let p = mat_to_personpose(0, &completed);
    let v: Vec<PersonPose> = vec![p];
    let s = save_poses_to_json(&v);
    let out_filename = "out/".to_owned() + filename;
    fs::write(out_filename, &s).expect("Unable to write file");
    //println!("{}",s);
}

#[test]
fn procrustes_works() {
    let a = Array::eye(3);
    let b = Array::eye(3);
    assert_eq!((procrustes(&a, &b).0[(0,0)] - 0.33).abs() < 0.1,true);
}

#[test]
fn load_json() {
    let ref_filename = "complete_pose_reference.json";
    let ref_contents = fs::read_to_string(ref_filename)
        .expect("Something went wrong reading the file");
    let ref_pose = load_poses_from_json(&ref_contents).unwrap();
    assert_eq!(ref_pose[0].keypoints3d[0][0] != 0.0 ,true);
}

#[test]
fn test_complete_w_reference(){
    let a = vec![PersonPose{id: 0, keypoints3d: [[0.0; 4]; 25]}];
    let b = vec![PersonPose{id: 0, keypoints3d: [[0.0; 4]; 25]}];
    let mask: Vec<bool> = (0..25).map(|x| a[0].keypoints3d[x][3]!=0.0).collect();
    let m_a = personpose_to_mat(&a[0], &mask);
    let (rotation, translation) = procrustes(&m_a, &m_a);
    //let completed = complete_with_transformed_ref(&a[0], &b[0], (&rotation, &translation));
}

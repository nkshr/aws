#include "stdafx.h"
// Copyright(c) 2012 Yohei Matsumoto, Tokyo University of Marine
// Science and Technology, All right reserved. 

// f_base is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// f_base is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with f_base.  If not, see <http://www.gnu.org/licenses/>. 

#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
using namespace std;

#define XMD_H
#ifdef SANYO_HD5400
#include <jpeglib.h>
#include <curl/curl.h>
#endif

#include "../util/aws_stdlib.h"
#include "../util/aws_thread.h"
#include "../util/c_clock.h"

#include <opencv2/opencv.hpp>
using namespace cv;

#include "f_misc.h"

////////////////////////////////////////////////////////// f_lcc members
f_lcc::f_lcc(const char * name): f_misc(name), m_ch_img_in(NULL), m_ch_img_out(NULL), 
	m_alpha(0.01), m_range(3.0), m_update_map(true)
{
	m_fmap[0] = '\0';

	register_fpar("ch_in", (ch_base**)&m_ch_img_in, typeid(ch_image_ref).name(), "Input image channel");
	register_fpar("ch_out", (ch_base**)&m_ch_img_out, typeid(ch_image_ref).name(), "Output image channel");
	
	register_fpar("flipx", &m_flipx, "Flip in x.");
	register_fpar("flipy", &m_flipy, "Flip in y.");

	register_fpar("alpha", &m_alpha, "Averaging coeffecient.");
	register_fpar("range", &m_range, "Clipping range.");

	register_fpar("fmap", m_fmap, 1024, "File path for map.");
	register_fpar("update", &m_update_map, "Update map.");
}

bool f_lcc::init_run()
{
	if(m_fmap[0]){
		FileStorage fs(m_fmap, FileStorage::READ);
		if(!fs.isOpened()){
			cerr << "Failed to open " << m_fmap << endl;
			m_update_map = true;			
		}else{
			FileNode fn = fs["LCCMap"];
			if(!fn.empty()){
				fn >> m_map;
			}
		}
	}else{
		m_update_map = true;
	}

	return true;
}

void f_lcc::destroy_run()
{
	if(m_update_map && m_fmap[0]){
		FileStorage fs(m_fmap, FileStorage::WRITE);
		fs << "LCCMap" << m_map;
	}
}

bool f_lcc::proc()
{
	long long t, i;
	Mat img = m_ch_img_in->get_img(t, i);

	if(img.empty())
		return true;

	if(m_t >= t){
		return true;
	}

	m_t = t;
	m_ifrm = i;
	if(m_map.empty() || m_map.size() != img.size()){
		m_map = Mat::zeros(img.rows, img.cols, CV_32FC2);
		m_update_map = true;
	}

	if(m_update_map){
		if(m_amap.empty() || m_vmap.empty() || m_dmap.empty() 
			|| m_amap.size() != img.size() 
			|| m_vmap.size() != img.size()
			|| m_dmap.size() != img.size()){
				if(m_update_map){
					m_amap.create(img.rows, img.cols, CV_32FC1);
					m_vmap.create(img.rows, img.cols, CV_32FC1);
					m_dmap.create(img.rows, img.cols, CV_32FC1);
					int num_pix = m_amap.cols * m_amap.rows;
					float * pa = m_amap.ptr<float>();
					float * pv = m_vmap.ptr<float>();
					float * pd = m_dmap.ptr<float>();
					float d = (float)(127. / m_range);
					float v = (float)(d * d);

					for(int i = 0; i < num_pix; i++){
						*pa = 127;
						*pv = v;
						*pd = d;
						pa++;
						pv++;
						pd++;
					}
				}
		}
	}

	if(m_flipx || m_flipy)
		awsFlip(img, m_flipx, m_flipy, false);

	// calcurate average and variance
	Mat img_out;
	switch(img.type()){
	case CV_16UC1:
		if(m_update_map)
			calc_avg_and_var_16uc1(img);
		img_out.create(img.rows, img.cols, CV_8UC1);
		filter_16uc1(img, img_out);
		break;
	case CV_8UC1:
		if(m_update_map)
			calc_avg_and_var_8uc1(img);
		img_out.create(img.rows, img.cols, CV_8UC1);
		filter_8uc1(img, img_out);
		break;
	case CV_8UC3:
		if(m_update_map)
			calc_avg_and_var_16uc3(img);
		img_out.create(img.rows, img.cols, CV_8UC3);
		filter_8uc3(img, img_out);
		break;
	case CV_16UC3:
		if(m_update_map)
			calc_avg_and_var_8uc3(img);
		img_out.create(img.rows, img.cols, CV_8UC3);
		filter_16uc3(img, img_out);
		break;
	}

	m_ch_img_out->set_img(img_out, t, i);

	return true;
}

void f_lcc::calc_avg_and_var_16uc1(Mat & img)
{
	unsigned short * pimg = img.ptr<unsigned short>();
	float * pavg = m_amap.ptr<float>();
	float * pvar = m_vmap.ptr<float>();
	float * pdev = m_dmap.ptr<float>();
	float * pm = m_map.ptr<float>();
	int num_pix = img.cols * img.rows;
	double ialpha = 1.0 - m_alpha;
	for(int i = 0; i < num_pix; i++){
		*pavg = (float)(m_alpha * *pimg +  ialpha * *pavg);
		double d = *pimg - *pavg;
		*pvar = (float)(m_alpha * d * d + ialpha * *pvar);
		*pdev = (float)sqrt(*pvar);

		double a, b;
		double rd = m_range * *pdev;
		a = *pavg - rd;
		b = *pavg + rd;
		pm[0] = (float)(255.0 / (2 * rd));
		pm[1] = (float)(- pm[0] * a);

		pavg++;
		pvar++;
		pdev++;
		pimg++;
		pm+=2;
	}
}

void f_lcc::calc_avg_and_var_8uc1(Mat & img)
{
	unsigned char * pimg = img.ptr<uchar>();
	float * pavg = m_amap.ptr<float>();
	float * pvar = m_vmap.ptr<float>();
	float * pdev = m_dmap.ptr<float>();
	float * pm = m_map.ptr<float>();
	int num_pix = img.cols * img.rows;
	double ialpha = 1.0 - m_alpha;
	for(int i = 0; i < num_pix; i++){
		*pavg = (float)(m_alpha * *pimg +  ialpha * *pavg);
		double d = *pimg - *pavg;
		*pvar = (float)(m_alpha * d * d + ialpha * *pvar);
		*pdev = (float)sqrt(*pvar);	

		double a, b;
		double rd = m_range * *pdev;
		a = *pavg - rd;
		b = *pavg + rd;
		pm[0] = (float)(255.0 / (2 * rd));
		pm[1] = (float)(- pm[0] * a);

		pavg++;
		pvar++;
		pdev++;
		pimg++;
		pm+=2;
	}
}

void f_lcc::calc_avg_and_var_16uc3(Mat & img)
{
	unsigned short * pimg = img.ptr<unsigned short>();
	float * pavg = m_amap.ptr<float>();
	float * pvar = m_vmap.ptr<float>();
	float * pdev = m_dmap.ptr<float>();
	float * pm = m_map.ptr<float>();
	int num_pix = img.cols * img.rows;
	double ialpha = 1.0 - m_alpha;
	for(int i = 0; i < num_pix; i++){
		double p = (double) (pimg[0] + pimg[1] + pimg[2]) * 0.333;
		*pavg = (float)(m_alpha * p +  ialpha * *pavg);
		double d = *pimg - *pavg;
		*pvar = (float)(m_alpha * d * d + ialpha * *pvar);
		*pdev = (float)sqrt(*pvar);

		double a, b;
		double rd = m_range * *pdev;
		a = *pavg - rd;
		b = *pavg + rd;
		pm[0] = (float)(255.0 / (2 * rd));
		pm[1] = (float)(- pm[0] * a);

		pavg++;
		pvar++;
		pdev++;
		pimg+=3;
		pm+=2;
	}
}

void f_lcc::calc_avg_and_var_8uc3(Mat & img)
{
	unsigned char * pimg = img.ptr<uchar>();
	float * pavg = m_amap.ptr<float>();
	float * pvar = m_vmap.ptr<float>();
	float * pdev = m_dmap.ptr<float>();
	float * pm = m_map.ptr<float>();
	int num_pix = img.cols * img.rows;
	double ialpha = 1.0 - m_alpha;
	for(int i = 0; i < num_pix; i++){
		double p = (double) (pimg[0] + pimg[1] + pimg[2]) * 0.333;
		*pavg = (float)(m_alpha * p +  ialpha * *pavg);
		double d = *pimg - *pavg;
		*pvar = (float)(m_alpha * d * d + ialpha * *pvar);
		*pdev = (float)sqrt(*pvar);

		double a, b;
		double rd = m_range * *pdev;
		a = *pavg - rd;
		b = *pavg + rd;
		pm[0] = (float)(255.0 / (2 * rd));
		pm[1] = (float)(- pm[0] * a);

		pavg++;
		pvar++;
		pdev++;
		pimg+=3;
		pm+=2;
	}
}


void f_lcc::filter_16uc1(Mat & in, Mat & out)
{
	unsigned short * pin = in.ptr<unsigned short>();
	unsigned char * pout = out.ptr<unsigned char>();
	int num_pix = in.cols * in.rows;
	float * pm = m_map.ptr<float>();
	for(int i = 0; i < num_pix; i++){
		*pout = saturate_cast<uchar>(pm[0] * *pin + pm[1]);
		pin++;
		pout++;
		pm+=2;
	}
}

void f_lcc::filter_8uc1(Mat & in, Mat & out)
{
	unsigned char * pin = in.ptr<unsigned char>();
	unsigned char * pout = out.ptr<unsigned char>();
	int num_pix = in.cols * in.rows;
	float * pm = m_map.ptr<float>();
	for(int i = 0; i < num_pix; i++){
		*pout = saturate_cast<uchar>(pm[0] * *pin + pm[1]);
		pin++;
		pout++;
		pm+=2;
	}
}

void f_lcc::filter_16uc3(Mat & in, Mat & out)
{
	unsigned short * pin = in.ptr<unsigned short>();
	unsigned char * pout = out.ptr<unsigned char>();
	int num_pix = in.cols * in.rows;
	float * pm = m_map.ptr<float>();
	for(int i = 0; i < num_pix; i++){
		pout[0] = saturate_cast<uchar>(pm[0] * pin[0] + pm[1]);
		pout[1] = saturate_cast<uchar>(pm[0] * pin[1] + pm[1]);
		pout[2] = saturate_cast<uchar>(pm[0] * pin[2] + pm[1]);
		pin+=3;
		pout+=3;
		pm+=2;
	}
}

void f_lcc::filter_8uc3(Mat & in, Mat & out)
{
	unsigned char * pin = in.ptr<unsigned char>();
	unsigned char * pout = out.ptr<unsigned char>();
	int num_pix = in.cols * in.rows;
	float * pm = m_map.ptr<float>();
	for(int i = 0; i < num_pix; i++){
		pout[0] = saturate_cast<uchar>(pm[0] * pin[0] + pm[1]);
		pout[1] = saturate_cast<uchar>(pm[0] * pin[1] + pm[1]);
		pout[2] = saturate_cast<uchar>(pm[0] * pin[2] + pm[1]);
		pin+=3;
		pout+=3;
		pm+=2;
	}
}

////////////////////////////////////////////////////////// f_stereo_disp members
const char * f_stereo_disp::m_str_out[IMG2 + 1] = {
	"disp", "img1", "img2"
};

f_stereo_disp::f_stereo_disp(const char * name): f_misc(name), m_ch_img1(NULL), m_ch_img2(NULL),
	m_ch_disp(NULL), m_bflipx(false), m_bflipy(false), m_bnew(false), m_bsync(false),
	m_bpl(false), m_bpr(false), m_bstp(false), m_brct(false),
	m_timg1(-1), m_timg2(-1), m_ifrm1(-1), m_ifrm2(-1), m_out(DISP)
{
	// channels
	register_fpar("ch_caml", (ch_base**)&m_ch_img1, typeid(ch_image_ref).name(), "Left camera channel");
	register_fpar("ch_camr", (ch_base**)&m_ch_img2, typeid(ch_image_ref).name(), "Right camera channel");
	register_fpar("ch_disp", (ch_base**)&m_ch_disp, typeid(ch_image_ref).name(), "Disparity image channel."); 

	register_fpar("out", (int*)&m_out, IMG2 + 1, m_str_out, "Output image type (for debug)");

	// file path for camera parameters
	register_fpar("fcpl", m_fcpl, 1024, "Camera parameter file of left camera.");
	register_fpar("fcpr", m_fcpr, 1024, "Camera parameter file of right camera.");
	register_fpar("fstp", m_fstp, 1024, "Stereo parameter file.");

	// flip flag
	register_fpar("flipx", &m_bflipx, "Flip image in x");
	register_fpar("flipy", &m_bflipy, "Flip image in y");

	// parameter for stereoSGBM
	register_fpar("update_bm", &m_sgbm_par.m_update, "Update BM parameter.");
	register_fpar("sgbm", &m_sgbm_par.m_bsg, "SGBM enabling flag.");
	register_fpar("minDisparity", &m_sgbm_par.minDisparity, "minDisparity for cv::StereoSGBM");
	register_fpar("numDisparities", &m_sgbm_par.numDisparities, "numDisparities for cv::StereoSGBM");
	register_fpar("blockSize", &m_sgbm_par.blockSize, "blockSize for cv::StereoSGBM");
	register_fpar("P1", &m_sgbm_par.P1, "P1 for cv::StereoSGBM");
	register_fpar("P2", &m_sgbm_par.P2, "P2 for cv::StereoSGBM");
	register_fpar("disp12MaxDiff", &m_sgbm_par.disp12MaxDiff, "disp12maxDiff for cv::StereoSGBM");
	register_fpar("preFilterCap", &m_sgbm_par.preFilterCap, "preFilterCap for cv::StereoSGBM");
	register_fpar("uniquenessRatio", &m_sgbm_par.uniquenessRatio, "uniquenessRatio for cv::StereoSGBM");
	register_fpar("speckleWindowSize", &m_sgbm_par.speckleWindowSize, "speckleWindowSize for cv::StereoSGBM");
	register_fpar("speckleRange", &m_sgbm_par.speckleRange, "speckleWindowSize for cv::StereoSGBM");
	register_fpar("mode", &m_sgbm_par.mode, "mode for cv::StereoSGBM");
}

f_stereo_disp::~f_stereo_disp()
{
}


bool f_stereo_disp::init_run()
{
	if(!m_ch_img1)
	{
		cerr << "Please set img1 channel." << endl;
		return false;
	}

	if(!m_ch_img2)
	{
		cerr << "Please set img2 channel." << endl;
		return false;
	}

	if(!m_ch_disp)
	{
		cerr << "Please set diparity image channel." << endl;
		return false;
	}

	if(!(m_bpl = m_camparl.read(m_fcpl))){
		cerr << "Failed to load camera1's intrinsic parameter." << endl;
		return false;
	}

	if(!(m_bpr = m_camparr.read(m_fcpr))){
		cerr << "Failed to load camera2's instrinsic parameter." << endl;
		return false;
	}

	if(!(m_bstp = load_stereo_pars())){
		cerr << "Failed to load stereo parameters." << endl;
		return false;
	}

	m_sgbm = StereoSGBM::create(m_sgbm_par.minDisparity, m_sgbm_par.numDisparities,
		m_sgbm_par.blockSize, m_sgbm_par.P1, m_sgbm_par.P2, m_sgbm_par.disp12MaxDiff, 
		m_sgbm_par.preFilterCap, m_sgbm_par.uniquenessRatio, m_sgbm_par.speckleWindowSize,
		m_sgbm_par.speckleRange, m_sgbm_par.mode);

	m_bm = StereoBM::create(m_sgbm_par.numDisparities, m_sgbm_par.blockSize);

	return true;
}

void f_stereo_disp::destroy_run()
{

}

bool f_stereo_disp::proc()
{
	long long timg1, timg2, ifrm1, ifrm2;
	Mat img1 = m_ch_img1->get_img(timg1, ifrm1);
	Mat img2 = m_ch_img2->get_img(timg2, ifrm2);

	if(img1.empty() || img2.empty())
		return true;

	if(m_timg1 != timg1){
		m_img1 = img1.clone();
		m_timg1 = timg1;
		m_ifrm1 = ifrm1;
		m_bnew = true;
	}

	if(m_timg2 != timg2){
		m_img2 = img2.clone();
		m_timg2 = timg2;
		m_ifrm2 = ifrm2;
		m_bnew = true;
	}

	if(m_ifrm1 == m_ifrm2)
		m_bsync =  true;

	if(!m_bsync || !m_bnew)
		return true;

	if(m_bpl && m_bpr && m_bstp && !m_brct){
		if(!(m_brct = rectify_stereo())){
			cerr << "Failed to rectify stereo images." << endl;
			return false;
		}
	}
	
	if(m_sgbm_par.m_update){
		m_sgbm->setBlockSize(m_sgbm_par.blockSize);
		m_sgbm->setDisp12MaxDiff(m_sgbm_par.disp12MaxDiff);
		m_sgbm->setMinDisparity(m_sgbm_par.minDisparity);
		m_sgbm->setMode(m_sgbm_par.mode);
		m_sgbm->setNumDisparities(m_sgbm_par.numDisparities);
		m_sgbm->setP1(m_sgbm_par.P1);
		m_sgbm->setP2(m_sgbm_par.P2);
		m_sgbm->setPreFilterCap(m_sgbm_par.preFilterCap);
		m_sgbm->setSpeckleRange(m_sgbm_par.speckleRange);
		m_sgbm->setSpeckleWindowSize(m_sgbm_par.speckleWindowSize);
		m_sgbm->setUniquenessRatio(m_sgbm_par.uniquenessRatio);
		m_sgbm_par.m_update = false;
	}

	awsFlip(m_img1, m_bflipx, m_bflipy, false);
	awsFlip(m_img2, m_bflipx, m_bflipy, false);

	remap(m_img1, m_img1, m_mapl1, m_mapl2, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	remap(m_img2, m_img2, m_mapr1, m_mapr2, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

	Mat disps16;
	if(m_sgbm_par.m_bsg)
		m_sgbm->compute(m_img1, m_img2, disps16);
	else
		m_bm->compute(m_img1, m_img2, disps16);

	disps16.convertTo(m_disp, CV_8U, 255 / (m_sgbm_par.numDisparities * 16.));

	switch(m_out){
	case DISP:
		m_ch_disp->set_img(m_disp, m_timg1, m_ifrm1);
		break;
	case IMG1:
		m_ch_disp->set_img(m_img1, m_timg1, m_ifrm1);
		break;
	case IMG2:
		m_ch_disp->set_img(m_img2, m_timg2, m_ifrm2);
		break;
	}

	m_bsync = false;
	m_bnew = false;
	return true;
}

bool f_stereo_disp::load_stereo_pars()
{
	if(!m_fstp[0]){
		cerr << "Please specify file of stereo parameters." << endl;
		return false;
	}
	FileStorage fs(m_fstp, FileStorage::READ);
	if(!fs.isOpened()){
		cerr << "Failed to open file " << m_fstp << endl;
		return false;
	}

	FileNode fn;

	fn = fs["Rlr"];
	if(fn.empty())
		return false;
	fn >> m_Rlr;

	fn = fs["Tlr"];
	if(fn.empty())
		return false;
	fn >> m_Tlr;

	fn = fs["E"];
	if(fn.empty())
		return false;
	fn >> m_E;

	fn = fs["F"];
	if(fn.empty())
		return false;
	fn >> m_F;

	fn = fs["Q"];
	if(fn.empty())
		return false;
	fn >> m_Q;

	return true;
}

bool f_stereo_disp::rectify_stereo()
{	
	if(!m_bpl || !m_bpr || !m_bstp)
		return false;
	if(m_img1.empty() || m_img2.empty())
		return false;

	Size sz(m_img1.cols, m_img1.rows);
	Mat Kl, Dl, Kr, Dr;
	if(m_camparl.isFishEye()){
		Kl = m_camparl.getCvPrjMat().clone();
		Dl = m_camparl.getCvDistFishEyeMat().clone();
		Kr = m_camparr.getCvPrjMat().clone();
		Dr = m_camparr.getCvDistFishEyeMat().clone();
		fisheye::stereoRectify(Kl, Dl, Kr, Dr, sz, m_Rlr, m_Tlr, 
			m_Rl, m_Rr, m_Pl, m_Pr, m_Q, 0);
		fisheye::initUndistortRectifyMap(Kl, Dl, m_Rl, m_Pl, sz, CV_16SC2, m_mapl1, m_mapl2);
		fisheye::initUndistortRectifyMap(Kr, Dr, m_Rr, m_Pr, sz, CV_16SC2, m_mapr1, m_mapr2);
	}else{
		Kl = m_camparl.getCvPrjMat().clone();
		Dl = m_camparl.getCvDistMat().clone();
		Kr = m_camparr.getCvPrjMat().clone();
		Dr = m_camparr.getCvDistMat().clone();
		stereoRectify(Kl, Dl, Kr, Dr, sz, m_Rlr, m_Tlr, 
			m_Rl, m_Rr, m_Pl, m_Pr, m_Q, CALIB_ZERO_DISPARITY, -1);
		initUndistortRectifyMap(Kl, Dl, m_Rl, m_Pl, sz, CV_16SC2, m_mapl1, m_mapl2);
		initUndistortRectifyMap(Kr, Dr, m_Rr, m_Pr, sz, CV_16SC2, m_mapr1, m_mapr2);		
	}
	return true;
}


////////////////////////////////////////////////////////// f_debayer members
const char * f_debayer::m_strBayer[UNKNOWN] = {
	"BG8", "GB8", "RG8", "GR8", "GR8NN", "GR8Q", "GR8GQ", "GR8DGQ", "BG16", "GB16", "RG16", "GR16"
};

bool f_debayer::proc(){
	long long timg;
	Mat img = m_pin->get_img(timg);
	if(m_timg == timg){
		return true;
	}
	m_timg = timg;
	Mat bgr;

	if(img.empty()){
		return true;
	}

	switch(m_type){
	case BG8:
		cnvBayerBG8ToBGR8(img, bgr);
		break;
	case GB8:
		cnvBayerGB8ToBGR8(img, bgr);
		break;
	case RG8:
		cnvBayerRG8ToBGR8(img, bgr);
		break;
	case GR8:
		cnvBayerGR8ToBGR8(img, bgr);
		break;
	case GR8NN:
		cnvBayerGR8ToBGR8NN(img, bgr);
		break;
	case GR8Q:
		cnvBayerGR8ToBGR8Q(img, bgr);
		break;
	case GR8GQ:
		cnvBayerGR8ToG8Q(img ,bgr);
		break;
	case GR8DGQ:
		cnvBayerGR8ToDG8Q(img, bgr);
		break;
	case BG16:
		cnvBayerBG16ToBGR16(img, bgr);
		break;
	case GB16:
		cnvBayerGB16ToBGR16(img, bgr);
		break;
	case RG16:
		cnvBayerRG16ToBGR16(img, bgr);
		break;
	case GR16:
		cnvBayerGR16ToBGR16(img, bgr);
		break;
	}

	m_pout->set_img(bgr, timg);
	return true;
}

////////////////////////////////////////////////////////// f_imread members
bool f_imread::proc()
{
	char buf[1024];
	bool raw = false;
	if(m_flist.eof()){
		cout << "File is finished." << endl;
		return false;
	}

	m_flist.getline(buf, 1024);
	char * d = NULL;
	char * u = NULL;

	{
		char * p = buf;
		for(;*p != '\0'; p++){
			if(*p == '.') d = p;
			if(*p == '_') u = p;
		}
		if(d == NULL){
			cerr << "The file does not have any extension." << endl;
			return true;
		}
		if(d[1] == 'r' && d[2] == 'a' && d[3] == 'w')
			raw = true;
	}

	Mat img;
	if(raw)
		read_raw_img(img, buf);
	else
		img = imread(buf);

	long long timg = get_time();
	if(u != NULL){
		*d = '\0';
		timg = atoll(u + 1);
	}

	if(m_verb){
		tmex tm;
		gmtimeex(timg / MSEC  + m_time_zone_minute * 60000, tm);
		snprintf(buf, 32, "[%s %s %02d %02d:%02d:%02d.%03d %d] ", 
			getWeekStr(tm.tm_wday), 
			getMonthStr(tm.tm_mon),
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			tm.tm_msec,
			tm.tm_year + 1900);
		cout << timg << "->" << buf << endl;
	}

	m_pout->set_img(img, timg);
	return true;
}


////////////////////////////////////////////////////////// f_imwrite members
const char * f_imwrite::m_strImgType[eitRAW+1] = {
	"tiff", "jpg", "jp2", "png", "raw"
};

bool f_imwrite::proc()
{
	long long timg;
	Mat img = m_pin->get_img(timg);

	if(m_cur_timg == timg || img.empty()) 
		return true;

	m_cur_timg = timg;

	char buf[1024];
	vector<int> param(2);
	snprintf(buf, 1024, "%s/%s_%lld.%s", m_path, m_name, timg, m_strImgType[m_type]);

	switch(m_type){
	case eitJP2:
	case eitTIFF:
		imwrite(buf, img);
		break;
	case eitJPG:
		param[0] = CV_IMWRITE_JPEG_QUALITY;
		param[1] = m_qjpg;
		imwrite(buf, img, param);
		break;
	case eitPNG:
		param[0] = CV_IMWRITE_PNG_COMPRESSION;
		param[1] = m_qpng;
		imwrite(buf, img, param);
		break;
	case eitRAW:
		write_raw_img(img, buf);
		break;
	}
	if(m_verb)
		cout << "Writing " << buf << endl;
	return true;
}

////////////////////////////////////////////////////////// f_gry member
bool f_gry::proc()
{
  ch_image * pin = dynamic_cast<ch_image*>(m_chin[0]);
  if(pin == NULL)
    return false;
  
  ch_image * pout = dynamic_cast<ch_image*>(m_chout[0]);
  if(pout == NULL)
    return false;
  
  ch_image * pclrout = dynamic_cast<ch_image*>(m_chout[1]);
  if(pclrout == NULL)
    return false;
  
  long long timg;
  Mat img = pin->get_img(timg);
  if(img.empty())
    return true;
  
  Mat out;
  cvtColor(img, out, CV_BGR2GRAY);
  
  pout->set_img(out, timg);
  pclrout->set_img(img, timg);
  return true;
}

////////////////////////////////////////////////////////// f_gauss members
bool f_gauss::cmd_proc(s_cmd & cmd)
{
	int num_args = cmd.num_args;
	char ** args = cmd.args;

	if(num_args < 2)
		return false;

	int itok = 2;

	if(strcmp(args[itok], "sigma") == 0){
		if(num_args != 4)
			return false;
		m_sigma = atof(args[itok+1]);
		return true;
	}

	return f_base::cmd_proc(cmd);
}


bool f_clip::cmd_proc(s_cmd & cmd)
{
	int num_args = cmd.num_args;
	char ** args = cmd.args;

	if(num_args < 2)
		return false;

	int itok = 2;

	if(strcmp(args[itok], "rc") == 0){
		if(num_args != 7)
			return false;
		m_rc_clip = Rect(atoi(args[itok+1]),
			atoi(args[itok+2]), atoi(args[itok+3]), atoi(args[itok+4]));

		return true;
	}

	return f_base::cmd_proc(cmd);
}

bool f_bkgsub::cmd_proc(s_cmd & cmd)
{
	int num_args = cmd.num_args;
	char ** args = cmd.args;

	if(num_args < 2)
		return false;

	int itok = 2;

	if(strcmp(args[itok], "rc") == 0){
		return true;
	}

	return f_base::cmd_proc(cmd);
	
}

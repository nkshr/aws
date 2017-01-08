#include "stdafx.h"
// Copyright(c) 2016 Yohei Matsumoto, Tokyo University of Marine
// Science and Technology, All right reserved. 

// f_state_estimator.cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// f_state_estimator.cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with f_state_estimator.cpp.  If not, see <http://www.gnu.org/licenses/>. 

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <thread>
#include <mutex>

using namespace std;

#include "../util/aws_stdlib.h"
#include "../util/aws_thread.h"
#include "../util/c_clock.h"

#include <opencv2/opencv.hpp>
using namespace cv;


#include "f_state_estimator.h"

f_state_estimator::f_state_estimator(const char * name) : f_base(name), m_ch_state(NULL), m_ch_estate(NULL),
m_tpos_prev(0), m_tvel_prev(0)

{
	m_Qx = Mat::zeros(2, 2, CV_32FC1);
	float * pQx = m_Qx.ptr<float>();
	m_Qv = Mat::zeros(2, 2, CV_32FC1);
	float * pQv = m_Qv.ptr<float>();
	m_Rx = Mat::zeros(2, 2, CV_32FC1);
	float * pRx = m_Rx.ptr<float>();
	m_Rv = Mat::zeros(2, 2, CV_32FC1);
	float * pRv = m_Rv.ptr<float>();
	
	register_fpar("ch_state", (ch_base**)(&m_ch_state), typeid(ch_state).name(), "State channel");
	register_fpar("ch_estate", (ch_base**)(&m_ch_state), typeid(ch_state).name(), "Estimated state channel");
	register_fpar("qxx", pQx, "Qx(0, 0)");
	register_fpar("qxy", pQx + 1, "Qx(0, 1)");
	register_fpar("qyx", pQx + 2, "Qx(1, 0) should be equal to Qx(0, 1)");
	register_fpar("qyy", pQx + 3, "Qx(1, 1)");
	register_fpar("quu", pQv, "Qv(0, 0)");
	register_fpar("quv", pQv + 1, "Qv(0, 1)");
	register_fpar("qvu", pQv + 2, "Qv(1, 0) should be equal to Qv(0, 1)");
	register_fpar("qvv", pQv + 3, "Qv(1, 1)");
	register_fpar("rxx", pQx, "Rx(0, 0)");
	register_fpar("rxy", pQx + 1, "Rx(0, 1)");
	register_fpar("ryx", pQx + 2, "Rx(1, 0) should be equal to Rx(0, 1)");
	register_fpar("ryy", pQx + 3, "Rx(1, 1)");
	register_fpar("ruu", pQv, "Rv(0, 0)");
	register_fpar("ruv", pQv + 1, "Rv(0, 1)");
	register_fpar("rvu", pQv + 2, "Rv(1, 0) should be equal to Rv(0, 1)");
	register_fpar("rvv", pQv + 3, "Rv(1, 1)");

}

f_state_estimator::~f_state_estimator()
{
}

bool f_state_estimator::init_run()
{
	if (!m_ch_state){
		cerr << "State channel is not connected." << endl;
		return false;
	}

	if (!m_ch_estate){
		cerr << "Estimated state channel is not connected." << endl;
		return false;
	}

	m_Px = Mat::zeros(2, 2, CV_32FC1);
	m_Pv = Mat::zeros(2, 2, CV_32FC1);
	return true;
}

void f_state_estimator::destroy_run()
{

}

bool f_state_estimator::proc()
{

	long long t = m_cur_time;
	// retrieve state from state channel
	long long tbih;
	float gps_lat, gps_lon, gps_alt, gps_galt;
	m_ch_state->get_position(tbih, gps_lat, gps_lon, gps_alt, gps_galt);
	long long tecef;
	float gps_xecef, gps_yecef, gps_zecef;
	m_ch_state->get_position_ecef(tecef, gps_xecef, gps_yecef, gps_zecef);
	long long tenu;
	Mat Renu = m_ch_state->get_enu_rotation(tenu);

	long long tvel;
	float cog, sog;
	m_ch_state->get_velocity(tvel, cog, sog);

	float dtx = (float)((m_cur_time - tbih) * (1.0 / (double)SEC));
	float dtv = (float)((m_cur_time - tvel) * (1.0 / (double)SEC));

	if (m_tpos_prev == 0){
		m_lat_prev = m_lat_opt = gps_lat;
		m_lon_prev = m_lon_opt = gps_lon;
		m_alt_prev = m_alt_opt = gps_alt;
		m_xecef_prev = m_xecef_opt = gps_xecef;
		m_yecef_prev = m_yecef_opt = gps_yecef;
		m_zecef_prev = m_zecef_opt = gps_zecef;
		m_Renu_prev = m_Renu_opt = Renu;
		m_ch_estate->set_pos(t, m_lat_opt, m_lon_opt, 0);
		m_ch_estate->set_pos_ecef(t, m_xecef_opt, m_yecef_opt, m_zecef_opt);
		m_ch_estate->set_enu_rot(t, m_Renu_opt);
		m_tpos_prev = t;
	}

	if (m_tvel_prev == 0){
		float cog_rad = (float)(cog * (PI / 180.));
		float cog_cos = (float)cos(cog_rad);
		float cog_sin = (float)sin(cog_rad);
		float sog_mps = (float)(sog * (1852. / 3600.));
		float u = (float)(sog_mps * cog_sin), v = (float)(sog_mps * cog_cos);
		m_cog_prev = cog;
		m_sog_prev = sog;
		m_u_prev = m_u_opt = u;
		m_v_prev = m_v_opt = v;

		m_tvel_prev = t;
	}

	// write estimated state to estate channel
	// estimation (x, y, u, v) is observed position in enu coordinate, (xp, yp, up, vp) is estimated position in enu coordinate
	//| xp |  =|  1  0  t  0  ||x|  = Fx(t) |x|
 	//| yp |   |  0  1  0  t  ||y|          |y|
	//                         |u|          |u| 
	//                         |v|          |v|
	//
	// constant velocity assumption
	//| up |  =|  1  0  ||u| = Fv(t) |u|
	//| vp |   |  0  1  ||v|         |v|

	// residual (xe, ye, ue, ve) is difference between  (xp, yp, up, vp) and  (x, y, u, v)
	// |ex| = |x| - |xe|
	// |ey|   |y|   |ye|
	//
	// |eu| = |u| - |ue|
	// |ev|   |v|   |ve|

	// State variable (xp, yp, up, vp) contains error q = (qx, qy, qu, qv), a 4-d normal random variables.
	// In unit time, covariance matrix Q is 
	// Qx = | qx  0 |
	//      |  0 qy |
	// Qv = | qu  0 |
	//      |  0 qv |

	// Predicted covariance matrices
	// Px(dtx) = dtx Qx + Fx(dtx) |Px(0)       0 | Fx(dtx)^t
	//                            |    0  Pv(dtv)|
	// Pv(dtv) = dtv Qv + Fv(dtv) Pv(0) Fv(dtv)^t
	// 
	//where Px(0) are Pv(0) covariances in previous estimations.
	// dtx and dtv are the elapsed times after last observations.

	// Ovservation errors 
	// Rx = | rx  0 |
	//      |  0 ry |
	// Rv = | ru  0 |
	//      |  0 rv |
	
	// Kalman Gain K is
	// Kx = Px(dtx)(Rx + P(dtx))^-1
	// Kv = Pv(dtv)(Rv + P(dtv))^-1

	// Estimation is done when observations take place.
	// | xe | = Kx | ex | + | x |
	// | ye |      | ey |   | y |
	//
	// | ue | = Kv | eu | +  | u |
	// | ve |      | ev |    | v |

	Mat Pdtv;
	Pdtv = dtv * m_Qv + m_Pv;
	if (tbih == tecef && tbih == tenu){
		Mat Pdtx;
		Pdtx = dtx * m_Qx + m_Px + dtx * dtx * Pdtv;

		// updating previous measurement
		if (tbih > m_tpos_prev){
			float xp = m_u_prev * dtx, yp = m_v_prev * dtx;
			Mat Kx = Pdtx * (m_Rx + Pdtx).inv();
			float x, y, z; // observed x, y, z
			eceftowrld(m_Renu_opt, m_xecef_opt, m_yecef_opt, m_zecef_opt, gps_xecef, gps_yecef, gps_zecef, x, y, z);
			float ex = (float)(x - xp), ey = (float)(y - yp);

			float *pK = Kx.ptr<float>(0);
			float xe, ye;

			xe = (float)(pK[0] * ex + pK[1] * ey + x);
			ye = (float)(pK[2] * ex + pK[3] * ey + y);
			wrldtoecef(m_Renu_opt, m_xecef_opt, m_yecef_opt, m_zecef_opt, xe, ye, 0.f, m_xecef_opt, m_yecef_opt, m_zecef_opt);
			eceftobih(m_xecef_opt, m_yecef_opt, m_zecef_opt, m_lat_opt, m_lon_opt, m_alt_opt);
			getwrldrotf(m_lat_opt, m_lon_opt, m_Renu_opt);

			m_ch_estate->set_pos(t, m_lat_opt, m_lon_opt, 0);
			m_ch_estate->set_pos_ecef(t, m_xecef_opt, m_yecef_opt, m_zecef_opt);
			m_ch_estate->set_enu_rot(t, m_Renu_opt);

			m_lat_prev = gps_lat;
			m_lon_prev = gps_lon;
			m_alt_prev = gps_alt;
			m_xecef_prev = gps_xecef;
			m_yecef_prev = gps_yecef;
			m_zecef_prev = gps_zecef;
			m_Renu_prev = Renu;
			m_tpos_prev = t;
		}
		else{
			// update prediction
			float xp = m_u_prev * dtx, yp = m_v_prev * dtx;
			float xpecef, ypecef, zpecef, plat, plon, palt;
			wrldtoecef(m_Renu_opt, m_xecef_opt, m_yecef_opt, m_zecef_opt, xp, yp, 0.f, xpecef, ypecef, zpecef);
			eceftobih(xpecef, ypecef, zpecef, plat, plon, palt);
			Mat R;
			getwrldrotf(plat, plon, R);
			m_ch_estate->set_pos(t, plat, plon, 0.f);
			m_ch_estate->set_pos_ecef(t, xpecef, ypecef, zpecef);
			m_ch_estate->set_enu_rot(t, R);
		}
	}

	if (tvel > m_tvel_prev){
		Mat Kv = Pdtv * (m_Rv + Pdtv).inv();
		float cog_rad = (float)(cog * (PI / 180.));
		float cog_cos = (float)cos(cog_rad);
		float cog_sin = (float)sin(cog_rad);
		float sog_mps = (float)(sog * (1852. / 3600.));
		float u = (float)(sog_mps * cog_sin), v = (float)(sog_mps * cog_cos);
		m_cog_prev = cog;
		m_sog_prev = sog;
		m_u_prev = u;
		m_v_prev = v;

		float eu = (float)(u - m_u_opt), ev = (float)(v - m_v_opt);
		float * pK = Kv.ptr<float>(0);
		m_u_opt = (float)(pK[0] * eu + pK[1] * ev + u);
		m_v_opt = (float)(pK[2] * eu + pK[3] * ev + v);
		m_tvel_prev = t;
	}

	return true;
}



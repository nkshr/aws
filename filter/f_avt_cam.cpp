// Copyright(c) 2013 Yohei Matsumoto, Tokyo University of Marine
// Science and Technology, All right reserved. 

// f_avt_cam.cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// f_avt_cam.cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with f_avt_cam.  If not, see <http://www.gnu.org/licenses/>. 

#include "stdafx.h"
#ifdef AVT_CAM
#include <cstdio>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>

#include <list>

#include "../util/aws_stdlib.h"
#include "../util/aws_thread.h"
#include "../util/c_clock.h"

using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include "f_avt_cam.h"

bool f_avt_cam::m_bready_api = false;

const char * f_avt_cam::strFrameStartTriggerMode[efstmUndef] = {
	"Freerun", "SyncIn1", "SyncIn2", "SyncIn3", "SyncIn4",
	"FixedRate", "Software"
};

// enum is defined in PvApi.h
const char * f_avt_cam::strPvFmt[ePvFmtBayer12Packed+1] = {
	"Mono8", "Mono16", "Bayer8", "Bayer16", "Rgb24", "Rgb48",
	"Yuv411", "Yuv422", "Yuv444", "Bgr24", "Rgba32", "Bgra32",
	"Mono12Packed",  "Bayer12Packed"
};

const char * f_avt_cam::strBandwidthCtrlMode[bcmUndef] = {
	"StreamBytesPerSecond", "SCPD", "Both"
};

const char * f_avt_cam::strExposureMode[emExternal+1] = {
	"Manual", "Auto", "AutoOnce", "External"
};

const char * f_avt_cam::strExposureAutoAlg[eaaFitRange+1] = {
	"Mean", "FitRange"
};

const char * f_avt_cam::strGainMode[egmExternal+1] = {
	"Manual", "Auto", "AutoOnce", "External"
};

const char * f_avt_cam::strWhitebalMode[ewmAutoOnce+1] = {
	"Manual", "Auto", "AutoOnce"
};

const char * f_avt_cam::m_strParams[NUM_PV_PARAMS] = {
	"host",	"nbuf", "PacketSize", "update", "FrameStartTriggerMode", "BandwidthCtrlMode", "StreamBytesPerSecond",
	"ExposureMode", "ExposureAutoAdjustTol", "ExposureAutoAlg", "ExposureAutoMax", "ExposureAutoMin",
	"ExposureAutoOutliers", "ExposureAutoRate", "ExposureAutoTarget", "ExposureValue", "GainMode",
	"GainAutoAdjustTol", "GainAutoMax", "GainAutoMin", "GainAutoOutliers", "GainAutoRate",
	"GainAutoTarget", "GainValue", "WhitebalMode", "WhitebalAutoAdjustTol", "WhitebalAutoRate",
	"WhitebalValueRed", "WhitebalValueBlue", "Height", "Width", "RegionX",
	"RegionY", "PixelFormat", "ReverseX", "ReverseY", "ReverseSoftware",
	"Strobe1Duration", "Strobe1Delay", "Strobe1ControlledDuration", "Strobe1Mode",
	"SyncOut1Mode", "SyncOut2Mode", "SyncOut3Mode", "SyncOut4Mode", 
	"SyncOut1Invert", "SyncOut2Invert", "SyncOut3Invert", "SyncOut4Invert", 
	"GvspResendPercent", "GvspRetries", "GvspTimeout",
	"fcp", "ud", "udw", "udh", "emsg", "verb"
};

const char * f_avt_cam::strStrobeMode[esmUndef] = {
		"AcquisitionTriggerReady", "FrameTriggerReady", "FrameTrigger", "Exposing", "FrameReadabout",
		"Imaging", "Acquiring", "SynqIn1", "SynqIn2"
};

const char * f_avt_cam::strStrobeControlledDuration[escdUndef] = {
	"On", "Off"
};

const char * f_avt_cam::strSyncOutMode[esomUndef] = {
		"GPO", "AcquisitionTriggerReady", "FrameTriggerReady", "FrameTrigger", "ExposingFrameReadout",
		"Acquiring", "SyncIn1", "SyncIn2", "Strobe1"
};

const char * f_avt_cam::strSyncOutInvert[esoiUndef] = {
	"On", "Off"
};

void _STDCALL f_avt_cam::s_cam_params::proc_frame(tPvFrame * pfrm)
{
	s_cam_params * pcam = (s_cam_params *) pfrm->Context[0];
	pcam->set_new_frm(pfrm); 
}

f_avt_cam::s_cam_params::s_cam_params(int icam): m_num_buf(5), 
	m_access(ePvAccessMaster), m_frame(NULL), m_PacketSize(8228),
	m_PixelFormat(__ePvFmt_force_32), m_update(false),
	m_FrameStartTriggerMode(efstmUndef),
	m_BandwidthCtrlMode(bcmUndef),
	m_StreamBytesPerSecond(0/*115000000*/), m_ExposureMode(emUndef), m_ExposureAutoAdjustTol(UINT_MAX /*5*/),
	m_ExposureAutoAlg(eaaUndef/*eaaMean*/), m_ExposureAutoMax(UINT_MAX/*500000*/), m_ExposureAutoMin(UINT_MAX/*1000*/),
	m_ExposureAutoOutliers(UINT_MAX/*0*/), m_ExposureAutoRate(UINT_MAX/*100*/), m_ExposureAutoTarget(UINT_MAX/*50*/),
	m_ExposureValue(UINT_MAX/*100*/), m_GainMode(egmUndef), m_GainAutoAdjustTol(UINT_MAX/*5*/), m_GainAutoMax(UINT_MAX/*30*/),
	m_GainAutoMin(UINT_MAX/*10*/), m_GainAutoOutliers(UINT_MAX/*0*/), m_GainAutoRate(UINT_MAX /*100*/), m_GainAutoTarget(UINT_MAX /*50*/),
	m_GainValue(UINT_MAX/*10*/),
	m_WhitebalMode(ewmAuto), m_WhitebalAutoAdjustTol(5), m_WhitebalAutoRate(100),
	m_WhitebalValueRed(0), m_WhitebalValueBlue(0),
	m_Height(UINT_MAX), m_RegionX(UINT_MAX), m_RegionY(UINT_MAX), m_Width(UINT_MAX),
  m_BinningX(UINT_MAX), m_BinningY(UINT_MAX), m_DecimationHorizontal(0), m_DecimationVertical(0), m_ReverseSoftware(false), m_ReverseX(false), m_ReverseY(false),
  m_Strobe1Mode(esmUndef), m_Strobe1ControlledDuration(escdUndef), m_Strobe1Duration(UINT_MAX), m_Strobe1Delay(UINT_MAX),
  m_SyncOut1Mode(esomUndef), m_SyncOut2Mode(esomUndef), m_SyncOut3Mode(esomUndef), m_SyncOut4Mode(esomUndef),
  m_SyncOut1Invert(esoiUndef), m_SyncOut2Invert(esoiUndef), m_SyncOut3Invert(esoiUndef), m_SyncOut4Invert(esoiUndef),
  m_GvspResendPercent(FLT_MAX), m_GvspRetries(UINT_MAX), m_GvspTimeout(UINT_MAX),
  bundist(false), bemsg(false), verb(false)
{
	if(icam == -1){
		strParams = m_strParams;
	}else{
	  int num_params = sizeof(m_strParams) / sizeof(char*);
		strParams = new const char*[num_params];
		for(int i = 0; i < num_params; i++){
			int len = (int) strlen(m_strParams[i]) + 2;
			char * ptr =  new char[len]; // number and termination character.
			snprintf(ptr, len, "%s%d", m_strParams[i], icam);
			strParams[i] = ptr;
		}
	}
}

f_avt_cam::s_cam_params::~s_cam_params()
{
  if(m_strParams != strParams)
    {
      for(int i = 0; i < sizeof(m_strParams) / sizeof(char*); i++){
	delete[] strParams[i];
      }
      delete[] strParams;
    }
}

f_avt_cam::f_avt_cam(const char * name): f_base(name), m_ttrig_int(30*MSEC), m_ttrig_prev(0)
{
	register_fpar("Ttrig", &m_ttrig_int, "Trigger interval. Only for FrameStartTriggerMode=Software.");
}

f_avt_cam::~f_avt_cam()
{
}

void f_avt_cam::register_params(s_cam_params & cam)
{
	register_fpar(cam.strParams[0], cam.m_host, 1024, "Network address of the camera to be opened.");
	register_fpar(cam.strParams[1], &cam.m_num_buf, "Number of image buffers.");
	register_fpar(cam.strParams[2], &cam.m_PacketSize, "Size of ethernet packet (default 8228)");
	register_fpar(cam.strParams[3], &cam.m_update, "Update flag.");
	register_fpar(cam.strParams[4], (int*)&cam.m_FrameStartTriggerMode, efstmUndef, strFrameStartTriggerMode, "Frame Start Trigger mode (default Freerun)");
	register_fpar(cam.strParams[5], (int*)&cam.m_BandwidthCtrlMode, bcmUndef, strBandwidthCtrlMode, "Bandwidth control mode (default StreamBytesPerSecond)");
	register_fpar(cam.strParams[6], &cam.m_StreamBytesPerSecond, "StreamBytesPerSecond (default 115000000)");

	// about exposure
	register_fpar(cam.strParams[7], (int*) &cam.m_ExposureMode, (int)(emExternal) + 1, strExposureMode, "ExposureMode");
	register_fpar(cam.strParams[8], &cam.m_ExposureAutoAdjustTol, "ExposureAutoAdjusttol (default 5)");
	register_fpar(cam.strParams[9], (int*)&cam.m_ExposureAutoAlg, (int)eaaFitRange, strExposureAutoAlg, "ExposureAutoAlg (default Mean)");
	register_fpar(cam.strParams[10], &cam.m_ExposureAutoMax, "ExposureAutoMax (default 500000us)");
	register_fpar(cam.strParams[11], &cam.m_ExposureAutoMin, "ExposureAutoMin (default 1000us)");
	register_fpar(cam.strParams[12], &cam.m_ExposureAutoOutliers, "ExposureAutoOutliers (default 0)");
	register_fpar(cam.strParams[13], &cam.m_ExposureAutoRate, "ExposureAutoRate (default 100)");
	register_fpar(cam.strParams[14], &cam.m_ExposureAutoTarget, "ExposureAutoTarget (default 50)");
	register_fpar(cam.strParams[15], &cam.m_ExposureValue, "ExposureValue (default 100us)");

	// about gain
	register_fpar(cam.strParams[16], (int*)&cam.m_GainMode, (int)(egmExternal) + 1, strGainMode, "GainMode (default Auto)");
	register_fpar(cam.strParams[17], &cam.m_GainAutoAdjustTol, "GainAutoAdjusttol (default 5)");
	register_fpar(cam.strParams[18], &cam.m_GainAutoMax, "GainAutoMax (default 30db)");
	register_fpar(cam.strParams[19], &cam.m_GainAutoMin, "GainAutoMin (default 5db)");
	register_fpar(cam.strParams[20], &cam.m_GainAutoOutliers, "GainAutoOutliers (default 0)");
	register_fpar(cam.strParams[21], &cam.m_GainAutoRate, "GainAutoRate (default 100)");
	register_fpar(cam.strParams[22], &cam.m_GainAutoTarget, "GainAutoTarget (default 50)");
	register_fpar(cam.strParams[23], &cam.m_GainValue, "GainValue (default 10db)");

	// about white balance
	register_fpar(cam.strParams[24], (int*)&cam.m_WhitebalMode, (int)ewmAutoOnce+1, strWhitebalMode, "WhitebalMode (default Auto)");
	register_fpar(cam.strParams[25], &cam.m_WhitebalAutoAdjustTol, "WhitebaAutoAdjustTol (percent, default 5)");
	register_fpar(cam.strParams[26], &cam.m_WhitebalAutoRate, "WhitebalAutoRate (percent, default 100)");
	register_fpar(cam.strParams[27], &cam.m_WhitebalValueRed, "WhitebalValueRed (percent, default 0)");
	register_fpar(cam.strParams[28], &cam.m_WhitebalValueBlue, "WhitebalValueBlue (percent, default 0)");

	// Image format
	register_fpar(cam.strParams[29], &cam.m_Height, "Height of the ROI (1 to Maximum Height)");
	register_fpar(cam.strParams[30], &cam.m_Width, "Width of the ROI(1 to Maximum Width)");
	register_fpar(cam.strParams[31], &cam.m_RegionX, "Top left x position of the ROI (0 to Maximum Camera Width - 1)");
	register_fpar(cam.strParams[32], &cam.m_RegionY, "Top left y position of the ROI (0 to Maximum Camera Height -1)");
	register_fpar(cam.strParams[33], (int*)&cam.m_PixelFormat, (int)(ePvFmtBayer12Packed+1), strPvFmt, "Image format.");
	register_fpar(cam.strParams[34], &cam.m_ReverseX, "Reverse horizontally. (default: no)");
	register_fpar(cam.strParams[35], &cam.m_ReverseY, "Reverse vertically. (default: no)");	
	register_fpar(cam.strParams[36], &cam.m_ReverseSoftware, "Reverse by software according to ReverseX and ReverseY flags.");

	// Related to camera parameter
	cam.fcp[0] = '\0';
	register_fpar(cam.strParams[37], &cam.m_Strobe1Duration, "Duration of Strobe 1 (usec)");
	register_fpar(cam.strParams[38], &cam.m_Strobe1Delay, "Delay of Strobe 1 (usec)");
	register_fpar(cam.strParams[39], (int*) &cam.m_Strobe1ControlledDuration, (int) escdUndef, strStrobeControlledDuration,  "Enabling control of strobe 1. On or Off.");
	register_fpar(cam.strParams[40], (int*) &cam.m_Strobe1Mode, (int) esmUndef, strStrobeMode, "Strobe Mode,");
	register_fpar(cam.strParams[41], (int*) &cam.m_SyncOut1Mode, (int) esomUndef, strSyncOutMode, "Sync Out Mode for SyncOut1.");
	register_fpar(cam.strParams[42], (int*) &cam.m_SyncOut2Mode, (int) esomUndef, strSyncOutMode, "Sync Out Mode for SyncOut2.");
	register_fpar(cam.strParams[43], (int*) &cam.m_SyncOut3Mode, (int) esomUndef, strSyncOutMode, "Sync Out Mode for SyncOut3.");
	register_fpar(cam.strParams[44], (int*) &cam.m_SyncOut4Mode, (int) esomUndef, strSyncOutMode, "Sync Out Mode for SyncOut4.");
	
	register_fpar(cam.strParams[45], (int*) &cam.m_SyncOut1Invert, (int) esoiUndef, strSyncOutInvert, "Sync Out Invert of SyncOut1");
	register_fpar(cam.strParams[46], (int*) &cam.m_SyncOut2Invert, (int) esoiUndef, strSyncOutInvert, "Sync Out Invert of SyncOut2");
	register_fpar(cam.strParams[47], (int*) &cam.m_SyncOut3Invert, (int) esoiUndef, strSyncOutInvert, "Sync Out Invert of SyncOut3");
	register_fpar(cam.strParams[48], (int*) &cam.m_SyncOut4Invert, (int) esoiUndef, strSyncOutInvert, "Sync Out Invert of SyncOut4");
	register_fpar(cam.strParams[49], &cam.m_GvspResendPercent, "Maximum ratio of missing packet to still resend is requested.");
	register_fpar(cam.strParams[50], &cam.m_GvspRetries, "Maximum number of resend requests.");
	register_fpar(cam.strParams[51], &cam.m_GvspTimeout, "The time till request resend.");

	register_fpar(cam.strParams[52], cam.fcp, 1024, "File path to the camera parameter. (AWS camera parameter format)");
	register_fpar(cam.strParams[53], &cam.bundist, "Enabling undistort.");
	register_fpar(cam.strParams[54], &cam.szud.width, "Width of the undistorted image.");
	register_fpar(cam.strParams[55], &cam.szud.height, "Height of the undistorted image.");
	register_fpar(cam.strParams[56], &cam.bemsg, "If asserted, error message is enabled in the callback.");
	register_fpar(cam.strParams[57], &cam.verb, "Verbose for debug.");
}

const char * f_avt_cam::get_err_msg(int code)
{
	const char * msg = f_base::get_err_msg(code);
	if(msg)
		return msg;

	switch(code){
	case FERR_AVT_CAM_INIT:
		return "Failed to initialize PvAPI.";
	case FERR_AVT_CAM_OPEN:
		return "Failed to open camera.";
	case FERR_AVT_CAM_ALLOC:
		return "Failed to allocate frame buffers.";
	case FERR_AVT_CAM_CLOSE:
		return "Failed to close camera.";
	case FERR_AVT_CAM_CFETH:
		return "Failed to reconfigure ethernet frame size.";
	case FERR_AVT_CAM_START:
		return "Failed to start capture.";
	case FERR_AVT_CAM_STOP:
		return "Failed to stop capture.";
	case FERR_AVT_CAM_CH:
		return "Channel is not set correctly.";
	case FERR_AVT_CAM_DATA:
		return "Incomplete image data.";
	}

	return NULL;
}


bool f_avt_cam::init_interface(){
	tPvErr err = PvInitialize();

	if(err != ePvErrSuccess){
		f_base::send_err(NULL, __FILE__, __LINE__, FERR_AVT_CAM_INIT);
		return false;
	}

	m_bready_api = true;
	return true;
}

void f_avt_cam::destroy_interface(){
	PvUnInitialize();
	m_bready_api = false;
}

bool f_avt_cam::s_cam_params::config_param()
{
	tPvErr err;

	if(m_PixelFormat == __ePvFmt_force_32){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "PixelFormat", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get PiexelFomrat" << endl;
			return false;
		}
		m_PixelFormat = getPvImageFmt(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "PixelFormat", strPvFmt[m_PixelFormat]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set PiexelFomrat" << endl;
			return false;
		}
	}

	err = PvAttrUint32Get(m_hcam, "SensorWidth", (tPvUint32*)&m_sWidth);
	if(err != ePvErrSuccess){
	  cerr << "Failed to get SensorWidth" << endl;
	  return false;
	}

	err = PvAttrUint32Get(m_hcam, "SensorHeight", (tPvUint32*)&m_sHeight);
	if(err != ePvErrSuccess){
	  cerr << "Failed to get SensorHeight" << endl;
	  return false;
	}

	if(m_Height == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "Height", (tPvUint32*)&m_Height);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Height" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "Height", (tPvUint32)m_Height);
		if(err != ePvErrSuccess){
			cerr << "failed to set Height" << endl;
			return false;
		}
	}

	if(m_Width == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "Width", (tPvUint32*)&m_Width);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Width" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "Width", (tPvUint32)m_Width);
		if(err != ePvErrSuccess){
			cerr << "failed to set Width" << endl;
			return false;
		}
	}

	if(m_RegionX == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "RegionX", (tPvUint32*)&m_RegionX);
		if(err != ePvErrSuccess){
			cerr << "Failed to get RegionX" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "RegionX", (tPvUint32)m_RegionX);
		if(err != ePvErrSuccess){
			cerr << "failed to set RegionX" << endl;
			return false;
		}
	}

	if(m_RegionY == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "RegionY", (tPvUint32*)&m_RegionY);
		if(err != ePvErrSuccess){
			cerr << "Failed to get RegionY" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "RegionY", (tPvUint32)m_RegionY);
		if(err != ePvErrSuccess){
			cerr << "failed to set RegionY" << endl;
			return false;
		}
	}

	if(m_BinningX == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "BinningX", (tPvUint32*)&m_BinningX);
		if(err != ePvErrSuccess){
			cerr << "Failed to get BinningX" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "BinningX", (tPvUint32)m_BinningX);
		if(err != ePvErrSuccess){
			cerr << "failed to set BinningX" << endl;
			return false;
		}
	}
	
	if(m_BinningY == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "BinningY", (tPvUint32*)&m_BinningY);
		if(err != ePvErrSuccess){
			cerr << "Failed to get BinningY" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "BinningY", (tPvUint32)m_BinningY);
		if(err != ePvErrSuccess){
			cerr << "failed to set BinningY" << endl;
			return false;
		}
	}

	/*
	if(m_DecimationHorizontal == 0){
		long long val;
		err = PvAttrInt64Get(m_hcam, "DecimationHorizontal", (tPvInt64*)&val);
		if(err != ePvErrSuccess){
			cerr << "Failed to get DecimationHorizontal" << endl;
			return false;
		}
		m_DecimationHorizontal = (int) val;
	}else{
		long long val = m_DecimationHorizontal;
		err = PvAttrInt64Set(m_hcam, "DecimationHorizontal", (tPvInt64)val);
		if(err != ePvErrSuccess){
			cerr << "failed to set DecimationHorizontal" << endl;
			return false;
		}
	}

	if(m_DecimationVertical == 0){
		long long val;
		err = PvAttrInt64Get(m_hcam, "DecimationVertical", (tPvInt64*)&val);
		if(err != ePvErrSuccess){
			cerr << "Failed to get DecimationVertical" << endl;
			return false;
		}
		m_DecimationVertical = (int) val;
	}else{
		long long val = m_DecimationVertical;
		err = PvAttrInt64Set(m_hcam, "DecimationVertical", (tPvInt64)val);
		if(err != ePvErrSuccess){
			cerr << "Failed to set DecimationVertical" << endl;
			return false;
		}
	}
	*/
	return config_param_dynamic();
}


bool f_avt_cam::s_cam_params::init(f_avt_cam * pcam, ch_base * pch)
{
	m_bactive = true;

	if(fcp[0] != '\0'){
		//loading camera parameter
		if(!cp.read(fcp)){
			cerr << "Failed to read camera parameter " << fcp << endl;
			fcp[0] = '\0';
			bundist = false;
		}
		
		if(bundist){
			R = Mat::eye(3, 3, CV_64FC1);
			if(cp.isFishEye()){
				fisheye::initUndistortRectifyMap(cp.getCvPrjMat(), 
					cp.getCvDistFishEyeMat(), 
					R, Pud, szud, CV_16SC2, udmap1, udmap2);
			}else{
				initUndistortRectifyMap(cp.getCvPrjMat(), cp.getCvDistMat(),
					R, Pud, szud, CV_16SC2, udmap1, udmap2);
			}
		}else{
			Pud = Mat();
			udmap1 = Mat();
			udmap2 = Mat();
			szud = Size(640, 480);
		}
	}


	int m_size_buf = 0;

	pout = dynamic_cast<ch_image_ref*>(pch);
 
	if(!pout){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_CH);
		return false;
	}

	if(pout){
		double * ptr = cp.getCvPrj();
		pout->set_int_campar(ECP_FX, ptr[0]);
		pout->set_int_campar(ECP_FY, ptr[1]);
		pout->set_int_campar(ECP_CX, ptr[2]);
		pout->set_int_campar(ECP_CY, ptr[3]);

		if(cp.isFishEye()){
			pout->set_fisheye(true);
			pout->set_int_campar(ECPF_K1, ptr[4]);
			pout->set_int_campar(ECPF_K2, ptr[5]);
			pout->set_int_campar(ECPF_K3, ptr[6]);
			pout->set_int_campar(ECPF_K4, ptr[7]);
		}else{
			pout->set_fisheye(false);
			pout->set_int_campar(ECP_K1, ptr[4]);
			pout->set_int_campar(ECP_K2, ptr[5]);
			pout->set_int_campar(ECP_P1, ptr[6]);
			pout->set_int_campar(ECP_P2, ptr[7]);
			pout->set_int_campar(ECP_K3, ptr[8]);
			pout->set_int_campar(ECP_K4, ptr[9]);
			pout->set_int_campar(ECP_K5, ptr[10]);
			pout->set_int_campar(ECP_K6, ptr[11]);
		}
	}
	tPvErr err;

	// init_interface shoudl be called before running pcam filter
	if(!m_bready_api){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_INIT);
		return false;
	}

	// opening camera by IP address
	unsigned long IpAddr = inet_addr(m_host);
	err = PvCameraOpenByAddr(IpAddr, m_access, &m_hcam);
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_OPEN);
		return false;
	}
	
	if(!config_param()){
		goto cam_close;
	}

	// getting frame size sent from camera
	err = PvAttrUint32Get(m_hcam, "TotalBytesPerFrame", (tPvUint32*)&m_size_buf);

	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_ALLOC);
		goto cam_close;
	}
	// allocating image buffer
	m_frame = new tPvFrame[m_num_buf];
	if(m_frame == NULL){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_ALLOC);
		goto cam_close;
	}

	memset(m_frame, 0, sizeof(tPvFrame) * m_num_buf);
	m_frm_done.resize(m_num_buf);

	m_mat_frame.resize(m_num_buf);
	for(int ibuf = 0; ibuf < m_num_buf; ibuf++){
		switch(m_PixelFormat){
		case ePvFmtMono8:
			m_mat_frame[ibuf] = Mat(m_Height, m_Width, CV_8UC1);
			break;
		case ePvFmtMono16:
			m_mat_frame[ibuf] = Mat(m_Height, m_Width,  CV_16UC1);
			break;
		case ePvFmtBayer8:
			m_mat_frame[ibuf] = Mat(m_Height, m_Width,  CV_8UC1);
			break;
		case ePvFmtBayer16:
			m_mat_frame[ibuf] = Mat(m_Height, m_Width,  CV_16UC1);
			break;
		case ePvFmtRgb24:
			m_mat_frame[ibuf] = Mat(m_Height, m_Width,  CV_8UC3);
			break;
		case ePvFmtRgb48:
			m_mat_frame[ibuf] = Mat(m_Height, m_Width,  CV_16UC3);
			break;
		case ePvFmtYuv411:
		case ePvFmtYuv422:
		case ePvFmtYuv444:
		case ePvFmtBgr24:
		case ePvFmtRgba32:
		case ePvFmtBgra32:
		case ePvFmtMono12Packed:
		case ePvFmtBayer12Packed:
		default:
			break;
		}
	}
#if  defined(_M_AMD64) || defined(_x64)
	unsigned long long ibuf;
#else
	unsigned int ibuf;
#endif

	for(ibuf = 0; ibuf < (unsigned) m_num_buf; ibuf++){
		m_frm_done[ibuf] = false;
		m_frame[ibuf].Context[0] = (void*) this;
		m_frame[ibuf].Context[1] = (void*) ibuf;
		m_frame[ibuf].ImageBufferSize = m_size_buf;
		m_frame[ibuf].ImageBuffer = (void*) m_mat_frame[ibuf].data;
		if(!m_frame[ibuf].ImageBuffer){
			f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_ALLOC);
			goto free_buf;
		}
	}

	err = PvCaptureAdjustPacketSize(m_hcam, m_PacketSize);
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_CFETH);
		goto free_buf;
	}

	err = PvCaptureStart(m_hcam);
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_START);
		switch(err){
		case ePvErrUnplugged:
			cout << "Camera is unplugged." << endl;
			break;
		case ePvErrResources:
			cout << "System resources are not available." << endl;
			break;
		case ePvErrBandwidth:
			cout << "Bandwidth is not sufficient to start capture stream." << endl;
			break;
		}
		goto free_buf;
	}

	m_cur_frm = 0;
	for(ibuf = 0; ibuf < (unsigned) m_num_buf; ibuf++){
		tPvErr PvErr = PvCaptureQueueFrame(m_hcam, &m_frame[ibuf], proc_frame);
		switch(PvErr){
		case ePvErrUnplugged:
			cout << "Camera is unplugged." << endl;
			break;
		case ePvErrBadSequence:
			cout << "Capture stream is not activated." << endl;
			break;
		case ePvErrQueueFull:
			cout << "The frame queue is full." << endl;
			break;
		default:
			m_frm_done[ibuf] = false;
			break;
		};
	}

	if(m_FrameStartTriggerMode == efstmUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "FrameStartTriggerMode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get FrameStartTriggerMode" << endl;
			goto free_buf;
		}
		m_FrameStartTriggerMode = getFrameStartTriggerMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "FrameStartTriggerMode",
			strFrameStartTriggerMode[m_FrameStartTriggerMode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set FrameStartTriggerMode" << endl;
			goto free_buf;
		}
	}

	err = PvAttrEnumSet(m_hcam, "AcquisitionMode", "Continuous");
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_START);
		goto free_buf;
	}

	err = PvCommandRun(m_hcam, "AcquisitionStart");
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_START);
		goto free_buf;
	}

	return true;

free_buf:
	cerr << "Freeing buffer." << endl;
	delete m_frame;
	m_frame = NULL;

cam_close:
	cerr << "Closing camera." << endl;
	err = PvCameraClose(m_hcam);
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_CLOSE);
	}
	return false;
}

void f_avt_cam::s_cam_params::destroy(f_avt_cam * pcam)
{
  if(m_frame == NULL) // camera conntection has already been destroied.
    return;

	tPvErr err;
	err = PvCommandRun(m_hcam, "AcquisitionStop");
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_STOP);
	}

#ifdef _WIN32
	Sleep(200);
#else
	sleep(1);
#endif

	err = PvCaptureQueueClear(m_hcam);
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_STOP);
	}

	err = PvCaptureEnd(m_hcam);
	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_STOP);
	}

	if(m_frame != NULL){
		delete[] m_frame;
		m_frame = NULL;
	}

	err = PvCameraClose(m_hcam);

	if(err != ePvErrSuccess){
		f_base::send_err(pcam, __FILE__, __LINE__, FERR_AVT_CAM_CLOSE);
	}
}

bool f_avt_cam::s_cam_params::config_param_dynamic()
{
	tPvErr err;
	if(!m_ReverseSoftware){
	  tPvBoolean val;
	  if(m_ReverseX)
	    val = 1;
	  else
	    val = 0;

	  err = PvAttrBooleanSet(m_hcam, "ReverseX", (tPvBoolean)val);
	  if(err != ePvErrSuccess){
	    cerr << "Failed to set ReverseX as " << (int) val << endl;
	    if(val)
	      m_ReverseSoftware = true;
	  }

	  if(m_ReverseY)
	    val = 1;
	  else
	    val = 0;

	  err = PvAttrBooleanSet(m_hcam, "ReverseY", (tPvBoolean)val);
	  if(err != ePvErrSuccess){
	    cerr << "Failed to set ReverseY as " << (int) val << endl;
	    if(val)
	      m_ReverseSoftware = true;
	  }
	}

	if(m_BandwidthCtrlMode == bcmUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "BandwidthCtrlMode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get BandwidthCtrlMode" << endl;
			return false;
		}
		m_BandwidthCtrlMode = getBandwidthCtrlMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "BandwidthCtrlMode", strBandwidthCtrlMode[m_BandwidthCtrlMode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set BandwidthCtrlMode" << endl;
			return false;
		}
	}

	if(m_StreamBytesPerSecond == 0){
		err = PvAttrUint32Get(m_hcam, "StreamBytesPerSecond", (tPvUint32*) &m_StreamBytesPerSecond);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Stream BytesPerSecond" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "StreamBytesPerSecond", (tPvUint32) m_StreamBytesPerSecond);
		if(err != ePvErrSuccess){
			cerr << "Failed to set StreamBytesPerSecond" << endl;
			return false;
		}
	}

	if(m_ExposureMode == emUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "ExposureMode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureMode" << endl;
			return false;
		}
		m_ExposureMode = getExposureMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "ExposureMode", strExposureMode[m_ExposureMode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureMode" << endl;
			return false;
		}
	}

	if(m_ExposureAutoAdjustTol == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureAutoAdjustTol", (tPvUint32*)&m_ExposureAutoAdjustTol);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureAutoAdjustTol" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureAutoAdjustTol", m_ExposureAutoAdjustTol);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoAdjustTol" << endl;
			return false;
		}
	}

	if(m_ExposureAutoAlg == eaaUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "ExposureAutoAlg", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureAutoAlg" << endl;
			return false;
		}
		m_ExposureAutoAlg = getExposureAutoAlg(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "ExposureAutoAlg", strExposureAutoAlg[m_ExposureAutoAlg]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoAlg" << endl;
			return false;
		}
	}

	if(m_ExposureAutoMax == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureAutoMax", (tPvUint32*)&m_ExposureAutoMax);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureAutomax" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureAutoMax", m_ExposureAutoMax);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoMax" << endl;
			return false;
		}
	}

	if(m_ExposureAutoMin == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureAutoMin", (tPvUint32*)&m_ExposureAutoMin);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureAutoMin" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureAutoMin", m_ExposureAutoMin);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoMin" << endl;
			return false;
		}
	}

	if(m_ExposureAutoOutliers == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureAutoOutliers", (tPvUint32*)&m_ExposureAutoOutliers);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureAutoOutliers" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureAutoOutliers", m_ExposureAutoOutliers);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoOutliers" << endl;
			return false;
		}
	}

	if(m_ExposureAutoRate == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureAutoRate", (tPvUint32*)&m_ExposureAutoRate);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureAutoRate" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureAutoRate", m_ExposureAutoRate);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoRate" << endl;
			return false;
		}
	}

	if(m_ExposureAutoTarget == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureAutoTarget", (tPvUint32*)&m_ExposureAutoTarget);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoTarget" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureAutoTarget", m_ExposureAutoTarget);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureAutoTarget" << endl;
			return false;
		}
	}

	if(m_ExposureValue == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "ExposureValue", (tPvUint32*)&m_ExposureValue);
		if(err != ePvErrSuccess){
			cerr << "Failed to get ExposureValue" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "ExposureValue", m_ExposureValue);
		if(err != ePvErrSuccess){
			cerr << "Failed to set ExposureValue" << endl;
			return false;
		}
	}

	if(m_GainMode == egmUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "GainMode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainMode" << endl;
			return false;
		}
		m_GainMode = getGainMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "GainMode", strGainMode[m_GainMode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainMode" << endl;
			return false;
		}
	}


	if(m_GainAutoAdjustTol == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainAutoAdjustTol", (tPvUint32*)&m_GainAutoAdjustTol);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainAutoAdjustTol" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainAutoAdjustTol", m_GainAutoAdjustTol);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainAutoAdjustTol" << endl;
			return false;
		}
	}

	if(m_GainAutoMax == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainAutoMax", (tPvUint32*)&m_GainAutoMax);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainAutoMax" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainAutoMax", m_GainAutoMax);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainAutoMax" << endl;
			return false;
		}
	}

	if(m_GainAutoMin == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainAutoMin", (tPvUint32*)&m_GainAutoMin);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainAutoMin" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainAutoMin", m_GainAutoMin);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainAutoMin" << endl;
			return false;
		}
	}

	if(m_GainAutoOutliers == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainAutoOutliers", (tPvUint32*)&m_GainAutoOutliers);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainAutoOutliers" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainAutoOutliers", m_GainAutoOutliers);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainAutoOutliers" << endl;
			return false;
		}
	}

	if(m_GainAutoRate == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainAutoRate", (tPvUint32*)&m_GainAutoRate);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainAutoRate" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainAutoRate", m_GainAutoRate);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainAutoRate" << endl;
			return false;
		}
	}

	if(m_GainAutoTarget == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainAutoTarget", (tPvUint32*)&m_GainAutoTarget);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainAutoTarget" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainAutoTarget", m_GainAutoTarget);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainAutoTarget" << endl;
			return false;
		}
	}

	if(m_GainValue == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "GainValue", (tPvUint32*)&m_GainValue);
		if(err != ePvErrSuccess){
			cerr << "Failed to get GainValue" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "GainValue", m_GainValue);
		if(err != ePvErrSuccess){
			cerr << "Failed to set GainValue" << endl;
			return false;
		}
	}

	if(m_Strobe1Mode == esmUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "Strobe1Mode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Strobe1Mode" << endl;
			return false;
		}
		m_Strobe1Mode = getStrobeMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "Strobe1Mode", strStrobeMode[m_Strobe1Mode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set Strobe1Mode" << endl;
			return false;
		}
	}

	if(m_Strobe1ControlledDuration == escdUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "Strobe1ControlledDuration", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Strobe1ControlledDuration" << endl;
			return false;
		}
		m_Strobe1ControlledDuration = getStrobeControlledDuration(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "Strobe1ControlledDuration", strStrobeControlledDuration[m_Strobe1ControlledDuration]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set Strobe1ControlledDuration" << endl;
			return false;
		}
	}

	if(m_Strobe1Duration == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "Strobe1Duration", (tPvUint32*)&m_Strobe1Duration);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Strobe1Duration" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "Strobe1Duration", m_Strobe1Duration);
		if(err != ePvErrSuccess){
			cerr << "Failed to set Strobe1Duration" << endl;
			return false;
		}
	}

	if(m_Strobe1Delay == UINT_MAX){
		err = PvAttrUint32Get(m_hcam, "Strobe1Delay", (tPvUint32*)&m_GainValue);
		if(err != ePvErrSuccess){
			cerr << "Failed to get Strobe1Delay" << endl;
			return false;
		}
	}else{
		err = PvAttrUint32Set(m_hcam, "Strobe1Delay", m_GainValue);
		if(err != ePvErrSuccess){
			cerr << "Failed to set Strobe1Delay" << endl;
			return false;
		}
	}

	if(m_SyncOut1Mode == esomUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut1Mode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut1Mode" << endl;
			return false;
		}
		m_SyncOut1Mode = getSyncOutMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut1Mode", strSyncOutMode[m_SyncOut1Mode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut1Mode" << endl;
			return false;
		}
	}

	if(m_SyncOut2Mode == esomUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut2Mode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut2Mode" << endl;
			return false;
		}
		m_SyncOut2Mode = getSyncOutMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut2Mode", strSyncOutMode[m_SyncOut2Mode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut2Mode" << endl;
			return false;
		}
	}

	if(m_SyncOut3Mode == esomUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut3Mode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut3Mode" << endl;
			return false;
		}
		m_SyncOut3Mode = getSyncOutMode(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut3Mode", strSyncOutMode[m_SyncOut3Mode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut3Mode" << endl;
			return false;
		}
	}

	if(m_SyncOut4Mode == esomUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut4Mode", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut4Mode" << endl;
		}else{
		  m_SyncOut4Mode = getSyncOutMode(buf);
		}
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut4Mode", strSyncOutMode[m_SyncOut4Mode]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut4Mode" << endl;
			return false;
		}
	}

	if(m_SyncOut1Invert == esoiUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut1Invert", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut1Invert" << endl;
			return false;
		}
		m_SyncOut1Invert = getSyncOutInvert(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut1Invert", strSyncOutInvert[m_SyncOut1Invert]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut1Invert" << endl;
			return false;
		}
	}

	if(m_SyncOut2Invert == esoiUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut2Invert", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut2Invert" << endl;
			return false;
		}
		m_SyncOut2Invert = getSyncOutInvert(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut2Invert", strSyncOutInvert[m_SyncOut2Invert]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut2Invert" << endl;
			return false;
		}
	}

	if(m_SyncOut3Invert == esoiUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut3Invert", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut3Invert" << endl;
			return false;
		}
		m_SyncOut3Invert = getSyncOutInvert(buf);
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut3Invert", strSyncOutInvert[m_SyncOut3Invert]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut3Invert" << endl;
			return false;
		}
	}

	if(m_SyncOut4Invert == esoiUndef){
		char buf[64];
		err = PvAttrEnumGet(m_hcam, "SyncOut4Invert", buf, 64, NULL);
		if(err != ePvErrSuccess){
			cerr << "Failed to get SyncOut4Invert" << endl;
		}else {
		  m_SyncOut4Invert = getSyncOutInvert(buf);
		}
	}else{
		err = PvAttrEnumSet(m_hcam, "SyncOut4Invert", strSyncOutInvert[m_SyncOut4Invert]);
		if(err != ePvErrSuccess){
			cerr << "Failed to set SyncOut4Invert" << endl;
			return false;
		}
	}

	if (m_GvspResendPercent == FLT_MAX){
		float val;
		err = PvAttrFloat32Get(m_hcam, "GvspResendPercent", &val);
		if (err != ePvErrSuccess){
			cerr << "Failed to get GvspResendPercent" << endl;
		}
		else{
			m_GvspResendPercent = val;
		}
	}
	else{
		err = PvAttrFloat32Set(m_hcam, "GvspResendPercent", m_GvspResendPercent);
		if (err != ePvErrSuccess){
			cerr << "Failed to set GvspResendPercent" << endl;
			return false;
		}
	}

	if (m_GvspRetries == UINT_MAX){
		tPvUint32 val;
		err = PvAttrUint32Get(m_hcam, "GvspRetries", &val);
		if (err != ePvErrSuccess){
			cerr << "Failed to get GvspRetries" << endl;
		}
		else{
			m_GvspRetries = val;
		}
	}
	else{
		err = PvAttrUint32Set(m_hcam, "GvspResendPercent", m_GvspRetries);
		if (err != ePvErrSuccess){
			cerr << "Failed to set GvspResendPercent" << endl;
			return false;
		}
	}

	if (m_GvspTimeout == UINT_MAX){
		tPvUint32 val;
		err = PvAttrUint32Get(m_hcam, "GvspTimeout", &val);
		if (err != ePvErrSuccess){
			cerr << "Failed to get GvspTimeout" << endl;
		}
		else{
			m_GvspTimeout = val;
		}
	}
	else{
		err = PvAttrUint32Set(m_hcam, "GvspTimeout", m_GvspTimeout);
		if (err != ePvErrSuccess){
			cerr << "Failed to set GvspTimeout" << endl;
			return false;
		}
	}

	return true;
}


void f_avt_cam::s_cam_params::set_new_frm(tPvFrame * pfrm)
{
	if(!m_bactive){
		return;
	}

#if  defined(_M_AMD64) || defined(x64)
	unsigned long long ibuf = *((unsigned long long *) (&pfrm->Context[1]));
#else
	unsigned int ibuf = *((unsigned int*) (&pfrm->Context[1]));
#endif

	if(pfrm->Status == ePvErrSuccess){
	  // cout << "Cam[" << m_host << "] Frame[" << pfrm->FrameCount <<"] Arrived " << endl;		
		if(!m_mat_frame[ibuf].empty()){
			Mat tmp;
			if(m_ReverseSoftware && (m_ReverseX || m_ReverseY)){
				awsFlip(m_mat_frame[ibuf], m_ReverseX, m_ReverseY, false);
			}
			if(bundist){
				remap(m_mat_frame[ibuf], tmp, udmap1, udmap2, INTER_LINEAR);
				tmp.copyTo(m_mat_frame[ibuf]);
			}
			m_cur_frm = pfrm->FrameCount;
			if (verb){
				cout << "Frame[" << m_cur_frm << "] arrived." << endl;
			}

			pout->set_img(m_mat_frame[ibuf], m_cur_time, m_cur_frm);
			pout->set_offset(m_RegionX, m_RegionY);
			pout->set_sz_sensor(m_sWidth, m_sHeight);
		}
	}else if(bemsg){
		switch(pfrm->Status){
		case ePvErrCancelled:
			cout << "Frame request is cancelled." << endl;
			break;
		case ePvErrDataMissing:
			cout << "Missing data." << endl;
			break;
		}
	}

	m_frm_done[ibuf] = true;

	tPvErr PvErr;
#if  defined(_M_AMD64) || defined(_x64)
	for(ibuf = 0; ibuf < (unsigned long long) m_num_buf; ibuf++){
#else
	for(ibuf = 0; ibuf < (unsigned int) m_num_buf; ibuf++){
#endif
		if(m_frm_done[ibuf]){
			if(m_mat_frame[ibuf].u->refcount == 1){
		//	if(!pout->is_buf_in_use((const unsigned char*) m_frame[ibuf].ImageBuffer)){
				PvErr = PvCaptureQueueFrame(m_hcam, &m_frame[ibuf], proc_frame);
				if(PvErr == ePvErrSuccess){
					m_frm_done[ibuf] = false;
				}else if(bemsg){
					switch(PvErr){
					case ePvErrUnplugged:
							cout << "Camera is unplugged." << endl;
						return;
					case ePvErrBadSequence:
							cout << "Capture stream is not activated." << endl;
						return;
					case ePvErrQueueFull:
							cout << "The frame queue is full." << endl;
						return;
					default:
						break;
					}
				}
			}
		}
	}
}

#endif

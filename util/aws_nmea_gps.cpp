#include "stdafx.h"

// Copyright(c) 2016 Yohei Matsumoto, All right reserved. 

// aws_nmea_gps.cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// aws_nmea_gps.cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with aws_nmea_gps.cpp.  If not, see <http://www.gnu.org/licenses/>. 
#include <cstdio>
#include <stdlib.h>
#include <wchar.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include "aws_sock.h"
#include "aws_thread.h"

using namespace std;

#include "aws_nmea.h"

/////////////////////////////////////////// gga decoder

bool c_gga::dec(const char * str)
{
	int i = 0;
	int ipar = 0;
	int len;
	char buf[32];
	char tok[32];
	while(ipar < 15){
		len = parstrcpy(buf, &str[i], ',');
		i += len + 1;
		if(len == 0){ 
			ipar++;
			continue;
		}

		switch(ipar){
		case 0: // $GPGGA 
			break;
		case 1: // TIME hhmmss
			parstrcpy(tok, buf, 2);
			m_h = (short) atoi(tok);
			parstrcpy(tok, buf+2, 2);
			m_m = (short) atoi(tok);
			parstrcpy(tok, buf+4, '\0');
			m_s = (float) atof(tok);
			break;
		case 2: // LAT
			parstrcpy(tok, buf, 2);
			m_lat_deg = atof(tok);
			parstrcpy(tok, buf+2, '\0');
			m_lat_deg += atof(tok) / 60;
			break;
		case 3: // N or S
			if(buf[0] == 'N')
				m_lat_dir = EGP_N;
			else
				m_lat_dir = EGP_S;
			break;				
		case 4: // LON
			parstrcpy(tok, buf, 3);
			m_lon_deg = atof(tok);
			parstrcpy(tok, buf + 3, '\0');
			m_lon_deg += atof(tok) / 60;
			break;
		case 5: // E or W
			if(buf[0] == 'E')
				m_lon_dir = EGP_E;
			else
				m_lon_dir = EGP_W;
			break;
		case 6: // Fix Stats
			m_fix = (e_gp_fix_stat)(buf[0] - '0');
		case 7: // NUM_SATS
			m_num_sats = atoi(buf);
			break;
		case 8: // HDOP
			m_hdop = (float) atof(buf);
			break;
		case 9: // Altitude
			m_alt = (float) atof(buf);
			break;
		case 10: // M
			break;
		case 11:// Geoidal separation
			m_geos = (float) atof(buf);
			break;
		case 12: // M
			break;
		case 13: // dgps age
			m_dgps_age = (float) atof(buf);
			break;
		case 14: // dgps station id
			if(parstrcpy(tok, buf, '*'))
				m_dgps_station = atoi(buf);
			break;
		}

		ipar++;
	}

	return true;
}

/////////////////////////////////////////// gsa decoder
bool c_gsa::dec(const char * str)
{
	int i = 0;
	int ipar = 0;
	int len;
	char buf[32];

	while(ipar < 18){
		len = parstrcpy(buf, &str[i], ',');
		i += len + 1;
		if(len == 0){ 
			ipar++;
			continue;
		}

		switch(ipar){
		case 0: // $GPRMC
			break;
		case 1: // Selected measurement mode
			if(buf[0] == 'A')
				s3d2d = 1;
			else if(buf[0] == 'M')
				s3d2d = 2;
			else
				s3d2d = 0;
			break;
		case 2: // Measurement mode
			if(buf[0] <'0' || buf[0] > '3')
				mm = 0;
			else
				mm = buf[0] - '0';			
			break;
		case 3: // sat1
		case 4: // sat2
		case 5: // sat3
		case 6: // sat4
		case 7: // sat5
		case 8: // sat6
		case 9: // sat7
		case 10: // sat8
		case 11: // sat9
		case 12: // sat10
		case 13: // sat11
		case 14: // sat12
			if(buf[0] == '\0'){
				sused[ipar - 3] = (unsigned short) atoi(buf);
			}
			break;
		case 15: // PDOP
			pdop = (float) atof(buf);
			break;
		case 16: // HDOP
			hdop = (float) atof(buf);
			break;
		case 17: // VDOP
			vdop = (float) atof(buf);
			break;
		}
		ipar++;
	}

	return true;
}

/////////////////////////////////////////// gsv decoder
bool c_gsv::dec(const char * str)
{
	int i = 0;
	int ipar = 0, npar = 4;
	int isat = 0;
	int len;
	char buf[32];

	while(ipar < npar){
		len = parstrcpy(buf, &str[i], ',');
		i += len + 1;
		if(len == 0){ 
			ipar++;
			continue;
		}

		switch(ipar){
		case 0: // $GPRMC
			break;
		case 1: // Number of sentences
			if(buf[0] < '0' || buf[0] > '4')
				ns = 0;
			else
				ns = buf[0] - '0';
			break;
		case 2: // Sentence index
			if(buf[0] < '0' || buf[0] > '4')
				si = 0;
			else
				si = buf[0] - '0';
			break;
		case 3: // Number of usable satellites 
			nsats_usable = atoi(buf);
			npar = nsats_usable - (si - 1) * 4;
			npar = min(4, npar);
			npar = npar * 4 + 4;
			isat = 0;
			break;
		case 4: // sat1
		case 8: // sat2
		case 12: // sat3
		case 16: // sat4
			sat[isat] = (unsigned short) atoi(buf);
			break;
		case 5: // elevation of sat1
		case 9: // elevation of sat2
		case 13: // elevation of sat3
		case 17: // elevation of sat4
			el[isat] = (unsigned short) atoi(buf);
			break;
		case 6: // azimuth of sat1
		case 10: // azimuth of sat2
		case 14: // azimuth of sat3
		case 18: // azimuth of sat4
			az[isat] = (unsigned short) atoi(buf);
			break;
		case 7: // sn
		case 11: // sn
		case 15: // sn
		case 19: // sn
			sn[isat] = (unsigned short) atoi(buf);
			isat++;
			break;
		}
		ipar++;
	}

	return true;
}

/////////////////////////////////////////// rmc decoder
bool c_rmc::dec(const char * str)
{
	int i = 0;
	int ipar = 0;
	int len;
	char buf[32];
	char tok[32];

	while(ipar < 12){
		len = parstrcpy(buf, &str[i], ',');
		i += len + 1;
		if(len == 0){ 
			ipar++;
			continue;
		}

		switch(ipar){
		case 0: // $GPRMC
			break;
		case 1: // TIME hhmmss
			parstrcpy(tok, buf, 2);
			m_h = (short) atoi(tok);
			parstrcpy(tok, buf+2, 2);
			m_m = (short) atoi(tok);
			parstrcpy(tok, buf+4, '\0');
			m_s = (float) atof(tok);
			break;
		case 2: // Validity flag
			if(buf[0] == 'A')
				m_v = true;
			else
				m_v = false;
			break;
		case 3: // Lat
			parstrcpy(tok, buf, 2);
			m_lat_deg = (float) atof(tok);
			parstrcpy(tok, buf+2, '\0');
			m_lat_deg += atof(tok) / 60;
			break;
		case 4: // N or S
			if(buf[0] == 'N')
				m_lat_dir = EGP_N;
			else
				m_lat_dir = EGP_S;
			break;				
		case 5: // LON
			parstrcpy(tok, buf, 3);
			m_lon_deg = (float) atof(tok);
			parstrcpy(tok, buf + 3, '\0');
			m_lon_deg += atof(tok) / 60;
			break;
		case 6: // E or W
			if(buf[0] == 'E')
				m_lon_dir = EGP_E;
			else
				m_lon_dir = EGP_W;
			break;
		case 7: // Speed
			m_vel = atof(buf);
			break;
		case 8: // Course
			m_crs = atof(buf);
			break;
		case 9: // Date
			parstrcpy(tok, buf, 2);
			m_dy = atoi(tok);
			parstrcpy(tok, buf+2, 2);
			m_mn = atoi(tok);
			parstrcpy(tok, buf+4, 2);
			m_yr = atoi(tok);
			break;
		case 10: // Course Variation
			m_crs_var = atof(buf);
			break;
		case 11: // Direction of Variation
			if(buf[0] == 'E')
				m_crs_var_dir = EGP_E;
			else
				m_crs_var_dir = EGP_W;
			break;
		}
		ipar++;
	}

	return true;
}

bool c_vtg::dec(const char * str)
{
	int i = 0;
	int ipar = 0;
	int len;
	char buf[32];

	while(ipar < 9){
		len = parstrcpy(buf, &str[i], ',');
		i += len + 1;
		if(len == 0){ 
			ipar++;
			continue;
		}

		switch(ipar){
		case 0: // $**VTG
			break;
		case 1: // course (True)
			crs_t = (float)atof(buf);
			break;
		case 2: // 'T'
			if(buf[0] != 'T')
				goto vtgerror;
			break;
		case 3: // course (Magnetic)
			crs_m = (float)atof(buf);
			break;
		case 4: // 'M'
			if(buf[0] != 'M')
				goto vtgerror;
			break;				
		case 5: // velocity (kts)
			v_n = (float)atof(buf);
			break;
		case 6: // 'N'
			if(buf[0] != 'N')
				goto vtgerror;
			break;
		case 7: // velocity (km/h)
			v_k = (float)atof(buf);
			break;
		case 8: // 'K'
			if(buf[0] != 'K')
				goto vtgerror;
			break;
		case 9: // Fix status
			if(buf[0] == 'N')
				fm = EGPF_LOST;
			else if(buf[0] == 'A')
				fm = EGPF_GPSF;
			else if(buf[0] == 'D')
				fm = EGPF_DGPSF;
			else if(buf[0] == 'E')
				fm = EGPF_ESTM;
			else
				goto vtgerror;
		}
		ipar++;
	}

	return true;
vtgerror:
	return false;
}

////////////////////////////////////////////////zda decoder
bool c_zda::dec(const char * str)
{
	int i = 0;
	int ipar = 0;
	int len;
	char buf[32];
	char tok[32];

	while(ipar < 7){
		len = parstrcpy(buf, &str[i], ',');
		i += len + 1;
		if(len == 0){ 
			ipar++;
			continue;
		}

		switch(ipar){
		case 0: // $GPRMC
			break;
		case 1: // TIME hhmmss
			parstrcpy(tok, buf, 2);
			m_h = (short) atoi(tok);
			parstrcpy(tok, buf+2, 2);
			m_m = (short) atoi(tok);
			parstrcpy(tok, buf+4, '\0');
			m_s = (float) atof(tok);
			break;
		case 2: // day
			m_dy = atoi(buf);
			break;
		case 3: // month
			m_mn = atoi(buf);
			break;
		case 4: // year
			m_yr = atoi(buf);
			break;				
		case 5: // local zone hour offset
			m_lzh = atoi(buf);
			break;
		case 6: // local zone minute offset
			m_lzm = atoi(buf);
			break;
		}
		ipar++;
	}

	return true;
}


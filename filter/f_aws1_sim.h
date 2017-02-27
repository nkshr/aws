// Copyright(c) 2014 Yohei Matsumoto, Tokyo University of Marine
// Science and Technology, All right reserved. 

// f_sample.cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Publica License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// f_sample.cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with f_sample.cpp.  If not, see <http://www.gnu.org/licenses/>. 

#ifndef _F_AWS1_SIM_H_
#define _F_AWS1_SIM_H_
#include "f_base.h"

struct s_aws1_mod{
  int res_meng;
  int res_rud;

  struct s_aws1_state{
    float ang_vel;  //angular velocity  over ground
    float sog; //speed over ground
    long long ot; //observation time 

    s_aws1_state();
    //s_aws1_state(const s_aws1_state &state) = delete;
  };

  s_aws1_state **table;

  s_aws1_mod();
  s_aws1_mod(const s_aws1_mod &mod) = delete;
  ~s_aws1_mod();

  s_aws1_mod::s_aws1_state** allocate2d();
  void init();

  void update(const unsigned char meng, const unsigned char rud,
	      const float ang_vel, const float sog, const long long ot);

  void interpolate_table();
  s_aws1_state& get_nearest_cell(const int meng, const int rud);

  //void reshape();
  
  void destroy();

  bool write(const char * fname);
  bool read(const char * fname);

  bool write_sog_csv(const char * fname);
  bool write_ang_vel_csv(const char * fname);
  bool write_ot_csv(const char * fname);

  s_aws1_state get_state(const unsigned char meng, const unsigned char rud);
};

class f_aws1_sim: public f_base 
{
 private:
  bool m_bwrite_ang_vel_csv;
  bool m_bwrite_sog_csv;
  bool m_bwrite_ot_csv;
  bool m_binterpolate;

  char m_fmod[1024];
  char m_fang_vel_csv[1024];
  char m_fsog_csv[1024];
  char m_fot_csv[1024];
  
  s_aws1_mod m_mod;
  s_aws1_ctrl_stat m_ctrl_stat;

  ch_aws1_ctrl_inst * m_ch_ctrl_ui;
  ch_aws1_ctrl_inst * m_ch_ctrl_ap1;
  ch_aws1_ctrl_inst * m_ch_ctrl_ap2;
  
  ch_aws1_ctrl_stat * m_ch_ctrl_stat; 
  ch_state * m_ch_state;

  void get_inst();

  void map_stat();

 public:
  f_aws1_sim(const char * fname);
  
  virtual bool init_run();
  
  virtual void destroy_run();
  
  virtual bool proc();
};

class f_aws1_mod: public f_base 
{
 private:
  bool m_bwrite_mod;
  bool m_bwrite_ang_vel_csv;
  bool m_bwrite_sog_csv;
  bool m_bwrite_ot_csv;
  bool m_binterpolate;
  bool m_breshape_mod;
  bool m_bverb;
  bool m_bupdate;

  char m_fmod[1024];
  char m_fang_vel_csv[1024];
  char m_fsog_csv[1024];
  char m_fot_csv[1024];

  int m_res_meng;
  int m_res_rud;
  int m_max_dstat_rud;
  int m_max_dstat_meng;

  //angular velocity and sog are supposed to be constant with respect to following parameters.
  //float m_max_dang_vel;
  float m_max_dsog;
  float m_max_dang_vel;
  float m_ref_ang_vel;
  float m_ref_sog;
  float m_prev_cog;
  
  long long m_ost; //observation start time
  long long m_min_ot; //minimum observation time
  long long m_prev_t;

  s_aws1_mod m_mod;

  s_aws1_ctrl_stat m_ref_stat;

  ch_aws1_ctrl_stat * m_ch_ctrl_stat;
  ch_state * m_ch_state;

  void reshape_mod();
 public:
  f_aws1_mod(const char * fname);
  
  virtual bool init_run();
  
  virtual void destroy_run();
  
  virtual bool proc();
    
};

#endif

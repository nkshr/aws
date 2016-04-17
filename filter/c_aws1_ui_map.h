// Copyright(c) 2016 Yohei Matsumoto, All right reserved. 

// c_aws1_ui_map.h is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// c_aws1_ui_map.h is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with c_aws1_ui_map.h.  If not, see <http://www.gnu.org/licenses/>. 

#ifndef _F_AWS1_UI_MAP_H_
#define _F_AWS1_UI_MAP_H_




// provides 2D map of objects and map near around 
class c_aws1_ui_map: public c_aws1_ui_core
{
protected:
	enum e_map_operation{
		EMO_EDT_WP, EMD_EDT_MAP
	} m_op;

	// EMD_EDT_WP: Edit way point
	//		A: Add waypoint to the cursor point
	//		B: Delete selected waypoint 
	//		Left/Right: Waypoint selection

	// EMD_EDT_MAP: Edit map (Still not defined)
	//

	Point2f m_ship_pts[3];
	Point2f m_circ_pts[36];

	float m_map_range; // range in meter (default 10km)
	Point2f m_cur_pos;
	Point2f m_map_pos; // relative position to my own ship (in meter)
public:
	c_aws1_ui_map(f_aws1_ui * _pui);

	virtual ~c_aws1_ui_map()
	{
	}

	virtual void js(const s_jc_u3613m  & js);
	virtual void draw(float xscale, float yscale);
	virtual void key(int key, int scancode, int action, int mods);
};

#endif

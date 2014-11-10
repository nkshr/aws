
#include <pthread.h>

class pthread_lock{
	pthread_mutex_t mtx;
public:
	pthread_lock(pthread_mutex_t & t):mtx(t){
		pthread_mutex_lock(&mtx);
	}

	~pthread_lock(){
		pthread_mutex_unlock(&mtx);
	}
};

#ifndef _UPTABLE_HPP_
#define _UPTABLE_HPP_

class Updatable {
public:
	virtual ~Updatable() {}
	// elapsed in secondss
	virtual void update(float elasped) = 0;
};

#endif
#ifndef _FPS_HPP_
#define _FPS_HPP_

#include <iostream>

class FPS {
public:
	FPS() {
		_cursor = 0;
		_full = false;
		for (int i = 0; i < 10; ++i) {
			_elapsed[i] = 0;
		}
	}

	// elapsed in seconds
	void update(float elapsed) {
		_elapsed[_cursor] = elapsed;
		_cursor = (_cursor + 1) % 10;
		_full |= (!_cursor);
	}

	void print() {
		float FPS = 0.0;
		if (!_full) {
			if (!_cursor) {
				return;
			}
			for (int i = 0; i < _cursor; ++i) {
				FPS += _elapsed[i];
			}
			FPS /= (float)_cursor;
			
		} else {
			for (int i = 0; i < 10; ++i) {
				FPS += _elapsed[i];
			}
			FPS /= 10.0;
		}
		std::cout << "FPS : " << 1.0 / FPS << std::endl;
	}

private:
	float _elapsed[10];
	int _cursor;
	bool _full;
};

#endif
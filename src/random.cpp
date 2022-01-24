#include <atomic>

#include <time.h>
#include <sys/time.h>

#include <random.hpp>
#include <math.hpp>


namespace rack {
namespace random {


thread_local Xoroshiro128Plus rng;

#if (defined __RX__) || (defined __sh__)
// Remove the atomic thread counter, since no multi-threading is going to be used for embedded
static uint64_t threadCounter = 0;
#else
static std::atomic<uint64_t> threadCounter {0};
#endif


void init() {
	// Don't reset state if already seeded
	if (rng.isSeeded())
		return;

	// Get epoch time in microseconds for seed
	struct timeval tv;
	gettimeofday(&tv, NULL);
	uint64_t usec = uint64_t(tv.tv_sec) * 1000 * 1000 + tv.tv_usec;
	// Add number of initialized threads so far to random seed, so two threads don't get the same seed if initialized at the same time.
	rng.seed(usec, threadCounter++);
	// Shift state a few times due to low seed entropy
	for (int i = 0; i < 4; i++) {
		rng();
	}
}


Xoroshiro128Plus& local() {
	return rng;
}


} // namespace random
} // namespace rack

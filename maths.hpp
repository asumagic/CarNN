#ifndef MATHS_HPP
#define MATHS_HPP

template <typename T>
T lerp(T a, T b, float t)
{
	return (1.f - t) * a + (t * b);
}

#endif // MATHS_HPP

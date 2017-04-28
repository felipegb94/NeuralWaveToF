# SimToF

A simulation engine for time of flight (ToF) imaging.

### Noise Model + Camera Parameters

The ToF simulator follows a physically accurate noise model and sensor saturation model. Once the correlation measurement of the modulation and demodulation functions is made, i.e we have a brightness value measured at a given pixl. If we denote this value as *B* the following noise/transformations are made:

1. **Add photon noise:** *B = B + sqrt(B)N(0,1)*. Where N(mu,sigmasq) is a normal random variable with mean, mu and variance sigmasq.
2. **Add read noise:** *B = B + sqrt(readNoiseSigmasq)N(0,1)*.
3. **Full well check:** There is a maximum value that can be measured at a given pixel (i.e the full well). If after adding photon and read noise our value exceeds the full well cap then, *B = np.maximum(np.minimum(B + noise, cam.fullWellCap),0)*. If `B < 0` then we set it to 0.
4. **Camera Gain:** `G = fullWellCap / (2^nBits)`. `B = B/G`
5. **Quantization Noise:** B = `np.round(B) / (2^nBits)`.

These 5 steps happen between lines 115 and 128.

### To dos

1. In Camera.py , figure out how to choose the focal length scaling factor.

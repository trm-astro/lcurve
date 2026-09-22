# Underlying physics

LCURVE is a code for modelling compact binaries, taking Roche geometry into account. It models each component (star1, star2, accretion disc, bright spot) as a grid of points where each point has a vector position and vector direction, a relative gravity, an area, and a flux. LCURVE is able to model much of the physics that is important for these compact binary systems such as Roche distortion, gravity darkening, limb darkening, gravitational lensing, Doppler beaming, reflection effect, and orbital period change. An rundown of the physics modelled by LCURVE and how each effect is modelled is explained here.

## Gravity darkening

Gravity darkening is the effect where regions of higher surface gravity on a star have higher pressures and therefore higher temperatures making them brighter. This effect is relevant for non-spherical stars such as those in close binaries experiencing Roche distortion. Gravity darkening is modelled as a power-law of effective temperature, $T_{eff}$, with the local gravity, $g$, where $\beta$ is the gravity darkening exponent (GDE). This is a bolometric quantity.

$$T_{eff}^{4} \propto g^{\beta}$$ (mymath2)
$$T_{eff} \propto g^{\beta/4}$$ (mymath2)

LCURVE models this by multiplying the point temperatures by $(g/g_{r})^{gravity_dark}$ when gdarkbol=True where $g$ is the gravity of the point being considered and $g_{r}$ is the reference gravity (i.e. the gravity for a non-distorted star). 

Filter-specific gravity darkening coefficients (GDCs) also exist, often given the symbol, $y$, and defined as

$$y = \frac{d \ln T_{eff}}{d \ln g}\left(\frac{\partial \ln I_{0}}{\partial \ln T_{eff}}\right)_{g} + \left(\frac{d \ln T_{eff}}{d \ln g}\right)_{T_{eff}}$$ (mymath2)

where

$$\frac{\beta}{4} = \frac{d \ln T_{eff}}{d \ln g}$$ (mymath2)

Since $\frac{d \ln T_{eff}}{d \ln g}$ is usually small,

$$\frac{\beta}{4} \approx \frac{y}{\left(\frac{\partial \ln I_{0}}{\partial \ln T_{eff}}\right)_{g}}$$ (mymath2)

When gdarkbol=False, LCURVE first does the above calculation to determine the equivalent bolometric GDE before multiplying the grid temperatures by $(g/g_{r})^{gravity_dark}$ as before.

## Limb darkening

## Gravitational lensing

## Doppler beaming

## Reflection effect

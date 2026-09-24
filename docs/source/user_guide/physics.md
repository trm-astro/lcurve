# Underlying physics

LCURVE is a code for modelling compact binaries, taking Roche geometry into account. It models each component (star1, star2, accretion disc, bright spot) as a grid of points where each point has a vector position and vector direction, a relative gravity, an area, and a flux. LCURVE is able to model much of the physics that is important for these compact binary systems such as Roche distortion, gravity darkening, limb darkening, gravitational lensing, Doppler beaming, reflection effect, and orbital period change. An rundown of the physics modelled by LCURVE and how each effect is modelled is explained here.

## Gravity darkening

Gravity darkening is the effect where regions of higher surface gravity on a star have higher pressures and therefore higher temperatures making them brighter. This effect is relevant for non-spherical stars such as those in close binaries experiencing Roche distortion. Gravity darkening is modelled as a power-law of effective temperature, $T_{eff}$, with the local gravity, $g$, where $\beta$ is the gravity darkening exponent (GDE). This is a bolometric quantity.

$$T_{\mathrm{eff}}^{4} \propto g^{\beta}$$ (equation1)
$$T_{\mathrm{eff}} \propto g^{\beta/4}$$ (equation2)

LCURVE models this by multiplying the point temperatures by $(g/g_{r})^{\texttt{gravity\_dark}}$ when gdark_bolom=True where $g$ is the gravity of the point being considered and $g_{r}$ is the reference gravity (i.e. the gravity for a non-distorted star). 

Filter-specific gravity darkening coefficients (GDCs) also exist, often given the symbol, $y$, and defined as

$$y(\lambda) = \frac{d \ln T_{\mathrm{eff}}}{d \ln g}\left(\frac{\partial \ln I_{0} (\lambda)}{\partial \ln T_{\mathrm{eff}}}\right)_{g} + \left(\frac{\partial \ln I_{0} (\lambda)}{\partial \ln g}\right)_{T_{\mathrm{eff}}}$$ (equation3)

where

$$\frac{\beta}{4} = \frac{d \ln T_{\mathrm{eff}}}{d \ln g}$$ (equation4)

Since $\left(\frac{\partial \ln I_{0} (\lambda)}{\partial \ln g}\right)_{T_{\mathrm{eff}}}$ is usually small,

$$\frac{\beta}{4} \approx \frac{y(\lambda)}{\left(\frac{\partial \ln I_{0}  (\lambda)}{\partial \ln T_{\mathrm{eff}}}\right)_{g}}$$ (equation5)

When gdark_bolom=False, LCURVE first does the above calculation to determine the equivalent bolometric GDE before multiplying the grid temperatures by $(g/g_{r})^{\texttt{gravity\_dark}}$ as before. Therefore, when using filter-specific GDCs, gdark_bolom must be set to False.

## Limb darkening

Limb darkening is the effect where the limb of the star appears darker as we are looking at that section of the stellar surface at a more oblique angle and therefore observing to a different depth in the photosphere with a different temperature and flux. Limb darkening can be modelled either as a polynomial

$$I(\mu) = 1 - \sum_{i=1}^{\leq 4} a_{i}(1 - \mu)^{i}$$

or using Claret's 4-parameter law

$$ I(\mu) = 1 - \sum_{i=1}^{4} a_{i}(1-\mu^{i/2})$$

The choice of law is set by the {py:attr}`limb1 <lcurve.Model.limb1>` and {py:attr}`limb2 <lcurve.Model.limb2>` parameters that can be set to either "Poly" or "Claret".

## Gravitational lensing

## Doppler beaming

Both doppler beaming and relativistic aberration are taken into account.

## Reflection effect

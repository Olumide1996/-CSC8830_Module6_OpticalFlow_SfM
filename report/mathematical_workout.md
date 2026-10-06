# CSC 8830 Module 6 — Mathematical Workouts

## Part A: Optical Flow, Motion Tracking, and Bilinear Interpolation

The assignment requires the derivation/description of the motion tracking equations, bilinear interpolation, and validation of the theoretical tracking result using two consecutive frames from each video. It also requires evidence from the computed optical flow. 

---

# 1. Optical Flow Formulation

## 1.1 Brightness Constancy Assumption

Optical flow estimates the apparent motion of image pixels between consecutive frames.

The basic assumption is that the brightness of a moving image point remains approximately constant over a very short time interval.

Let the image intensity be:

\[
I(x,y,t)
\]

where:

- \(x\) = horizontal image coordinate
- \(y\) = vertical image coordinate
- \(t\) = time
- \(u\) = horizontal image velocity
- \(v\) = vertical image velocity

If a point moves by \((u\Delta t,v\Delta t)\) during a small time interval \(\Delta t\), brightness constancy gives:

\[
I(x,y,t)
=
I(x+u\Delta t,\ y+v\Delta t,\ t+\Delta t)
\]

---

# 2. Taylor Expansion

Using a first-order Taylor expansion around \((x,y,t)\):

\[
I(x+u\Delta t,y+v\Delta t,t+\Delta t)
\]

is approximated by:

\[
I(x,y,t)
+
I_xu\Delta t
+
I_yv\Delta t
+
I_t\Delta t
\]

where:

\[
I_x=\frac{\partial I}{\partial x}
\]

\[
I_y=\frac{\partial I}{\partial y}
\]

\[
I_t=\frac{\partial I}{\partial t}
\]

Substituting into the brightness-constancy equation gives:

\[
I
=
I+
I_xu\Delta t+
I_yv\Delta t+
I_t\Delta t
\]

Subtracting \(I\) from both sides:

\[
0=
I_xu\Delta t+
I_yv\Delta t+
I_t\Delta t
\]

Dividing by \(\Delta t\):

\[
\boxed{
I_xu+I_yv+I_t=0
}
\]

This is the **optical flow constraint equation**.

---

# 3. Interpretation of the Optical Flow Constraint

The equation

\[
I_xu+I_yv+I_t=0
\]

relates:

- spatial intensity change in the \(x\)-direction,
- spatial intensity change in the \(y\)-direction,
- temporal intensity change,
- horizontal motion \(u\),
- vertical motion \(v\).

The equation provides one constraint for two unknown velocity components, \(u\) and \(v\).

This creates the **aperture problem**: motion cannot generally be determined uniquely from a single local image gradient.

Additional spatial or temporal information is therefore required to estimate motion robustly.

In this project, dense optical flow was calculated using the Farneback method, while individual image features were tracked using pyramidal Lucas-Kanade optical flow.

---

# 4. Frame-to-Frame Motion Tracking

Let a tracked image point in frame \(t\) be:

\[
p_t=
\begin{bmatrix}
x_t\\
y_t
\end{bmatrix}
\]

Suppose the optical-flow vector at that point is:

\[
d_t=
\begin{bmatrix}
u_t\\
v_t
\end{bmatrix}
\]

The predicted location in the next frame is:

\[
p_{t+1}^{predicted}
=
p_t+d_t
\]

Therefore:

\[
\boxed{
x_{t+1}^{predicted}=x_t+u_t
}
\]

\[
\boxed{
y_{t+1}^{predicted}=y_t+v_t
}
\]

The predicted location can then be compared with a separately tracked location obtained using Lucas-Kanade tracking.

---

# 5. Tracking Error

Let the optical-flow prediction be:

\[
p_{t+1}^{predicted}
=
(x_p,y_p)
\]

and let the Lucas-Kanade tracked point be:

\[
p_{t+1}^{actual}
=
(x_a,y_a)
\]

The Euclidean tracking error is:

\[
\boxed{
E=
\sqrt{
(x_p-x_a)^2+
(y_p-y_a)^2
}
}
\]

The error is measured in pixels.

A smaller value indicates stronger agreement between the dense optical-flow prediction and the Lucas-Kanade point-tracking result.

For this project, a point was classified as a **high-confidence inlier** when:

\[
\boxed{
E\leq1.0\text{ pixel}
}
\]

Points with:

\[
E>1.0\text{ pixel}
\]

were classified as outliers for the robust summary.

---

# 6. Bilinear Interpolation

Dense optical flow produces a motion vector at image-grid locations.

However, a tracked feature can occur at a non-integer, or **subpixel**, location.

For this reason, the flow vector can be estimated at the subpixel position using bilinear interpolation.

Suppose the desired location is:

\[
(x,y)
\]

with:

\[
x=x_0+a
\]

\[
y=y_0+b
\]

where:

\[
0\leq a<1
\]

and

\[
0\leq b<1
\]

The four neighboring flow vectors are:

\[
F_{00}=F(x_0,y_0)
\]

\[
F_{10}=F(x_0+1,y_0)
\]

\[
F_{01}=F(x_0,y_0+1)
\]

\[
F_{11}=F(x_0+1,y_0+1)
\]

The bilinear interpolation equation is:

\[
\boxed{
F(x,y)=
(1-a)(1-b)F_{00}
+
a(1-b)F_{10}
+
(1-a)bF_{01}
+
abF_{11}
}
\]

Because optical flow is a two-dimensional vector, the equation is applied to both the horizontal component \(u\) and vertical component \(v\).

Thus:

\[
F(x,y)=
\begin{bmatrix}
u(x,y)\\
v(x,y)
\end{bmatrix}
\]

---

# 7. Numerical Bilinear Interpolation Example

A real frame pair from the traffic video was used:

```text
Video: cars_traffic_30s.mp4
Frame t: 750
Frame t+1: 751
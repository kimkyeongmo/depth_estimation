#version 120
varying vec2 TexCoord;
uniform sampler2D screenTexture;
uniform float K1;
uniform float K2;

void main() {
    //set mid
    vec2 uv = TexCoord - vec2(0.5, 0.5);
    //distance calculate
    float r2 = dot(uv, uv);
    //distortion calculate (brown's distortion)
    vec2 distorted = uv * (1.0 + K1 * r2 + K2 * r2 * r2);
    //texture coordinate mapping
    distorted += vec2(0.5, 0.5);
    gl_FragColor = texture2D(screenTexture, distorted);
}
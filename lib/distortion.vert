#version 120
varying vec2 TexCoord;
void main(void){
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    TexCoord = gl_MultiTexCoord0.xy;
}
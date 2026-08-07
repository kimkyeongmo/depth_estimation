#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
//#include <windows.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>



namespace py = pybind11;


static GLFWwindow* g_window = nullptr;
static GLuint g_texture = 0;
static GLuint g_vao = 0, g_vbo = 0;
static GLuint g_shader = 0;
static cudaGraphicsResource_t g_cuda_resource = nullptr;

// Vertex Shader
static const char* VERT_SRC = R"glsl(
#version 460 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
)glsl";

// Fragment Shader
static const char* FRAG_SRC = R"glsl(
#version 460 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform float K1;
uniform float K2;
void main(){
    vec2 center = vUV.x < 0.5 ? vec2(0.25, 0.5) : vec2(0.75, 0.5);
    vec2 uv = vUV - center;
    uv.x *= 2.0;
    
    float r2 = dot(uv, uv);
    vec2 distorted = uv * (1.0 + K1 * r2 + K2 * r2 * r2);
    
    distorted.x /= 2.0;
    distorted += center;
    
    FragColor = texture(uTex, distorted);
}
)glsl";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
}


void init_window(int width, int height) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    g_window = glfwCreateWindow(width, height, "Output", nullptr, nullptr);
    glfwSetWindowPos(g_window,0,0);
    glfwMakeContextCurrent(g_window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    cudaGraphicsGLRegisterImage(&g_cuda_resource, g_texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard);
    float verts[] = {
        -1.f,-1.f, 0.f,1.f,
         1.f,-1.f, 1.f,1.f,
         1.f, 1.f, 1.f,0.f,
        -1.f,-1.f, 0.f,1.f,
         1.f, 1.f, 1.f,0.f,
        -1.f, 1.f, 0.f,0.f,
    };
    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &g_vbo);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    GLuint vs = compile_shader(GL_VERTEX_SHADER, VERT_SRC);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
    g_shader = glCreateProgram();
    glAttachShader(g_shader, vs);
    glAttachShader(g_shader, fs);
    glLinkProgram(g_shader);
    glDeleteShader(vs);
    glDeleteShader(fs);
    glfwMakeContextCurrent(nullptr);
}

void show_tensor(torch::Tensor tensor) {
    glfwMakeContextCurrent(g_window);
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Tensor must be float32");

    tensor = tensor.contiguous();

    int H = tensor.size(0);
    int W = tensor.size(1);

    cudaGraphicsMapResources(1, &g_cuda_resource, 0);
    cudaArray_t cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&cuda_array, g_cuda_resource, 0, 0);

    cudaMemcpy2DToArray(
        cuda_array, 0, 0,
        tensor.data_ptr<float>(),
        W * 4 * sizeof(float),
        W * 4 * sizeof(float),
        H,
        cudaMemcpyDeviceToDevice
    );

    cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g_shader);
    glUniform1f(glGetUniformLocation(g_shader, "K1"), -0.0f);
    glUniform1f(glGetUniformLocation(g_shader, "K2"), -0.0f);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glBindVertexArray(g_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(g_window);
    glfwPollEvents();
    glfwMakeContextCurrent(nullptr);
}

bool should_close() {
    return glfwWindowShouldClose(g_window);
}

void cleanup() {
    cudaGraphicsUnregisterResource(g_cuda_resource);
    glfwDestroyWindow(g_window);
    glfwTerminate();
}

PYBIND11_MODULE(CudaRuntime1, m) {
    m.def("init_window", &init_window);
    m.def("show_tensor", &show_tensor);
    m.def("should_close", &should_close);
    m.def("cleanup", &cleanup);
}
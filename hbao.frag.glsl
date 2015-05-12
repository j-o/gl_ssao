/* 
Based on DeinterleavedTexturing sample by Louis Bavoil
https://github.com/NVIDIAGameWorks/D3DSamples/tree/master/samples/DeinterleavedTexturing

*/

#version 430

// The pragma below is critical for optimal performance
// in this fragment shader to let the shader compiler
// fully optimize the maths and batch the texture fetches
// optimally

#pragma optionNV(unroll all)

#define M_PI 3.14159265f

// tweakables
const float  NUM_STEPS = 4;
const float  NUM_DIRECTIONS = 8; // texRandom/g_Jitter initialization depends on this

layout(location=0) uniform vec2 g_Float2Offset;
layout(location=1) uniform vec4 g_Jitter;
layout(location=2) uniform float g_depthLayer;
layout(binding=0) uniform sampler2DArray texArrayLinearDepth;
layout(binding=1) uniform sampler2D texViewNormal;

layout(location=3)  uniform float   RadiusToScreen;        // radius
layout(location=5)  uniform float   NegInvR2;     // -1 / (radius * radius)
layout(location=6)  uniform float   NDotVBias;

layout(location=8)  uniform vec2    InvQuarterResolution;

layout(location=9)  uniform float   AOMultiplier;
layout(location=10) uniform float   PowExponent;

layout(location=11) uniform vec4    projInfo;
layout(location=12) uniform int     projOrtho;

layout(location=0,index=0) out vec4 out_Color;

in vec2 texCoord;

//----------------------------------------------------------------------------------

vec3 UVToView(vec2 uv, float eye_z)
{
  return vec3((uv * projInfo.xy + projInfo.zw) * (projOrtho != 0 ? 1. : eye_z), eye_z);
}

vec3 FetchQuarterResViewPos(vec2 UV)
{
  float ViewDepth = textureLod(texArrayLinearDepth,vec3(UV, g_depthLayer),0).x;
  return UVToView(UV, ViewDepth);
}

//----------------------------------------------------------------------------------
float Falloff(float DistanceSquare)
{
  // 1 scalar mad instruction
  return DistanceSquare * NegInvR2 + 1.0;
}

//----------------------------------------------------------------------------------
// P = view-space position at the kernel center
// N = view-space normal at the kernel center
// S = view-space position of the current sample
//----------------------------------------------------------------------------------
float ComputeAO(vec3 P, vec3 N, vec3 S)
{
  vec3 V = S - P;
  float VdotV = dot(V, V);
  float NdotV = dot(N, V) * 1.0/sqrt(VdotV);

  // Use saturate(x) instead of max(x,0.f) because that is faster on Kepler
  return clamp(NdotV - NDotVBias,0,1) * clamp(Falloff(VdotV),0,1);
}

//----------------------------------------------------------------------------------
vec2 RotateDirection(vec2 Dir, vec2 CosSin)
{
  return vec2(Dir.x*CosSin.x - Dir.y*CosSin.y,
              Dir.x*CosSin.y + Dir.y*CosSin.x);
}

//----------------------------------------------------------------------------------
vec4 GetJitter()
{
  return g_Jitter;
}

//----------------------------------------------------------------------------------
float ComputeCoarseAO(vec2 FullResUV, float RadiusPixels, vec4 Rand, vec3 ViewPosition, vec3 ViewNormal)
{
  RadiusPixels /= 4.0;

  // Divide by NUM_STEPS+1 so that the farthest samples are not fully attenuated
  float StepSizePixels = RadiusPixels / (NUM_STEPS + 1);

  const float Alpha = 2.0 * M_PI / NUM_DIRECTIONS;
  float AO = 0;

  for (float DirectionIndex = 0; DirectionIndex < NUM_DIRECTIONS; ++DirectionIndex)
  {
    float Angle = Alpha * DirectionIndex;

    // Compute normalized 2D direction
    vec2 Direction = RotateDirection(vec2(cos(Angle), sin(Angle)), Rand.xy);

    // Jitter starting sample within the first step
    float RayPixels = (Rand.z * StepSizePixels + 1.0);

    for (float StepIndex = 0; StepIndex < NUM_STEPS; ++StepIndex)
    {
      vec2 SnappedUV = round(RayPixels * Direction) * InvQuarterResolution + FullResUV;
      vec3 S = FetchQuarterResViewPos(SnappedUV);

      RayPixels += StepSizePixels;

      AO += ComputeAO(ViewPosition, ViewNormal, S);
    }
  }

  AO *= AOMultiplier / (NUM_DIRECTIONS * NUM_STEPS);
  return clamp(1.0 - AO * 2.0,0,1);
}

//----------------------------------------------------------------------------------
void main()
{
  vec2 base = floor(gl_FragCoord.xy) * 4.0 + g_Float2Offset;
  vec2 uv = base * (InvQuarterResolution / 4.0);

  vec3 ViewPosition = FetchQuarterResViewPos(uv);
  vec4 NormalAndAO =  texelFetch( texViewNormal, ivec2(base), 0);
  vec3 ViewNormal =  -(NormalAndAO.xyz * 2.0 - 1.0);

  // Compute projection of disk of radius R into screen space
  float RadiusPixels = RadiusToScreen / (projOrtho != 0 ? 1.0 : ViewPosition.z);

  // Get jitter vector for the current full-res pixel
  vec4 Rand = GetJitter();

  float AO = ComputeCoarseAO(uv, RadiusPixels, Rand, ViewPosition, ViewNormal);

  out_Color = vec4(pow(AO, PowExponent), ViewPosition.z, 0, 0);
}


/*-----------------------------------------------------------------------
  Copyright (c) 2014, NVIDIA. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Neither the name of its contributors may be used to endorse 
     or promote products derived from this software without specific
     prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/

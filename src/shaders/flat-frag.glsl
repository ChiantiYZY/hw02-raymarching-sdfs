#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform vec4 u_Size;

in vec2 fs_Pos;
out vec4 out_Col;

vec3 mod289(vec3 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x)
{
  return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade(vec3 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}
vec2 fade2(vec2 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise2(vec2 P)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy = abs(gx) - 0.5 ;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;  
  g01 *= norm.y;  
  g10 *= norm.z;  
  g11 *= norm.w;  

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade2(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

// Classic Perlin noise, periodic variant
float pnoise(vec3 P, vec3 rep)
{
  vec3 Pi0 = mod(floor(P), rep); // Integer part, modulo period
  vec3 Pi1 = mod(Pi0 + vec3(1.0), rep); // Integer part + 1, mod period
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}


float impulse (float k, float x)
{
      float h = k * x;
      return h * exp(1.f - h);
}
mat4 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    );
}
mat4 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, c, s, 0),
        vec4(0, -s, c, 0),
        vec4(0, 0, 0, 1)
    );
}
float trianlgeWave(float x, float freq, float amplitude)
{
    //x = m - abs(i % (2*m) - m)

    //return x - abs(float(int(freq) % int(2.f * x)) - x);

    //y = abs((x++ % 6) - 3);
    //return float(abs((float((int(x)+1) % 6)) - 3.f));
    return float(abs(int(x * freq) % int(amplitude) - int((0.5 * amplitude))));
}







float opSmoothSubtraction( float d1, float d2, float k ) 
{
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); 

    return max(-d2, d1);
}
float opSmoothUnion( float d1, float d2, float k )
{
	float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
	return mix( d2, d1, h ) - k*h*(1.0-h);
}
float opSmoothIntersection( float d1, float d2, float k )
{
	float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
	return mix( d2, d1, h ) + k*h*(1.0-h);
}
vec3 opTwist( vec3 p )
{
    float  c = cos(10.0*p.y+10.0);
    float  s = sin(10.0*p.y+10.0);
    mat2   m0 = mat2(1.f, -1.f, 1.f, 1.f);
    mat2   m = mat2(c,-s,s,c);

    return vec3(m * p.xz,p.y);
}
vec3 opCheapBend( vec3 p )
{
    float step = smoothstep(-u_Size.w, u_Size.w, cos(u_Time * 0.05 + 3.f * p.x) * 0.01);
    float c = sin(step);
    float s = sin(step);
    mat2  m = mat2(c,-s,s,c);
    vec2 r = m * p.xy;
    vec3  q = vec3(r.xy, p.z);
    return q;
}

float onion( in float d, in float h )
{
    return abs(d)-h;
}



float sphere(vec3 p, float r)
{
    return length(p) - r;
}
float sdHexPrism( vec3 p, vec2 h )
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}
float sdCone( vec3 p, vec2 c )
{
    // c must be normalized
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}
float sdRoundCone( vec3 p, float r1, float r2, float h )
{
    vec2 q = vec2( length(p.xz), p.y );
    
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
    return dot(q, vec2(a,b) ) - r1;
}
float sdModPolar(inout vec2 p, float repetitions) {
    float angle = 2.*3.14/repetitions;
    float a = atan(p.y, p.x) + angle/2.;
    float r = length(p);
    float c = floor(a/angle);
    a = mod(a,angle) - angle/2.;
    p = vec2(cos(a), sin(a))*r;
    // For an odd number of repetitions, fix cell index of the cell in -x direction
    // (cell index would be e.g. -5 and 5 in the two halves of the cell):
    if (abs(c) >= (repetitions/2.)) c = abs(c);
    return c;
}


//displacement
vec2 obj1(vec3 pos)
{

    pos = pos + vec3(-7.5f, 2.5, -2.f);

    pos = pos / u_Size.x;
    float d = 1e10;
    float d1 = sphere(pos, 1.f);
    //float noise = noise(pos);
    float d1_mix = mix(d1, d1 + 0.3 * sin(10.f * pos.x)*sin(8.f * pos.y)*sin(6.f * pos.z), cos(u_Time * 0.01));
    d = min(d, d1_mix);
    

    return vec2(d * u_Size.x, 1.f);
}

//orbit
vec2 obj2(vec3 pos)
{
    pos = pos + vec3(7.5f, 2.5f, -2.f);

    pos = pos / 2.f;

    mat4 r = rotateZ(3.14 / 3.f);
    mat4 rr = rotateZ(-3.14 / 3.f);
    vec3 posR = vec3(r * vec4(pos, 1.f));
    vec3 posRR = vec3(rr * vec4(pos, 1.f));
    float d = 1e10;
    float d2 = sdTorus(pos, vec2(2.f, 0.05f));
    float d2_1 = sdTorus(posR , vec2(2.f, 0.05f));
    float d2_2 = sdTorus(posRR, vec2(2.f, 0.05f));   

    vec3 displace = vec3(2.f * sin(u_Time * 3.14 * u_Size.y), 0.f, 2.f * cos(u_Time * 3.14 * u_Size.y));
    vec3 displaceR = vec3(2.f * sin(-u_Time * 3.14 * u_Size.y), 0.f, 2.f * cos(-u_Time * 3.14 * u_Size.y));
    float d3 = sphere(pos + displaceR, 0.25); 
    float d3_1 = sphere(posR + displace, 0.25);
    float d3_2 = sphere(posRR + displace, 0.25);

    d = min(d, opSmoothUnion(d2_1, d2, 0.05));
    d = min(d, opSmoothUnion(d, d2_2, 0.05));
    d = min(d, opSmoothUnion(d, d3, 0.05));
    d = min(d, opSmoothUnion(d, d3_1, 0.05));
    d = min(d, opSmoothUnion(d, d3_2, 0.05));

    return vec2(d * 2.f, 2.f);
}

//roundbox
vec2 obj5(vec3 pos)
{
    float d = 1e10;

    pos = pos + vec3(7.5, -11.f, -2.0);

    float scale = mix(2.f, 1.f, abs(sin(u_Time * 0.01)));
    pos /= 3.f;
    
    float d1 = sdRoundBox(pos, vec3(1.f), 0.05);
    
	  float d2 = sphere(pos, 1.4);

    float ds = min(d, opSmoothSubtraction(d2, d1, 0.1));
    d = min(d, ds);

    return vec2(d * 2.f, 3.f);

} 

//heart
vec2 obj3(vec3 pos)
{
    float d = 1e10;
    pos = pos + vec3(7.5, -10.f, -2.0);
    float scale = impulse(u_Size.z, abs(sin(u_Time * 3.14 * 0.03)));
    pos /= 3.f * scale;

    mat4 r = rotateZ(3.14 / 6.f);
    mat4 rr = rotateZ(-3.14 / 6.f);

    float d1 = sdRoundCone(vec3(r * vec4(pos, 1.f)), 0.1, 0.5 , 1.0);
    float d2 = sdRoundCone(vec3(rr * vec4(pos, 1.f)), 0.1, 0.5 , 1.0);

    d = min(d, opSmoothUnion(d1, d2, 0.1));

    return vec2(d * scale , 3.f);

}

//wave
vec2 obj4(vec3 pos)
{
    float d = 1e10;
    // pos += vec3(-7.5, -12.5f + trianlgeWave(sin(u_Time * 0.01), 0.2f, 3.f), 0.f);
    pos += vec3(-9.f, -12.5f , 0.f);
    //pos *= 2.f;
    float d1 = sdBox(opCheapBend(pos), vec3(8.f, 0.3f, .5f));

    d = min(d, d1);

    return vec2(d, 4.f);

}



//torus
vec2 obj6(vec3 pos)
{
    pos = pos + vec3(-7.5, -11.f, -2.0);

    pos = pos / 5.f;

    vec3 pos1 = vec3(rotateX(3.14159 / 2.f) * vec4(pos, 1.f));
    vec3 pos2 = opTwist(pos);
    float d = 1e10;


    //float d1 = sdTorus(pos1, vec2(1.f, 0.1));
    float d2 = sdTorus(pos2, vec2(0.3f, 0.1));

    //float d_mix = mix(d1, d2, abs(sin(u_Time * 0.01)));

    d = min(d, d2);

    return vec2(d * 5.f, 6.f);

}

//shelf
vec2 obj8(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(0.f, -5.f, 0.f), vec3(15.f, .1f, 4.f));
    float d2 = sdBox(pos + vec3(0.f, 10.f, 0.f), vec3(15.f, .1f, 4.f));
    float d3 = sdBox(pos + vec3(0.f, -20.f, 0.f), vec3(15.f, .1f, 4.f));
float d4 = sdBox(pos + vec3(0.f, -5.f, 0.f), vec3(.1f, 15.f, 4.f));
    float d5 = sdBox(pos + vec3(-15.f, -5.f, 0.f), vec3(.1f, 15.f, 4.f));
    float d6 = sdBox(pos + vec3(15.f, -5.f, 0.f), vec3(.1f, 15.f, 4.f));
    


    d = min(d, d1);
    d = min(d, d2);
    d = min(d, d3);
     d = min(d, opSmoothUnion(d, d4, 0.01));
    d = min(d, opSmoothUnion(d, d5, 0.01));
    d = min(d, opSmoothUnion(d, d6, 0.01));
    

    return vec2(d, 8.f);
}

vec2 obj7(vec3 pos)
{
    float d = 1e10;
    
    float d7 = sdBox(pos + vec3(0.f, -5.f, 4.f), vec3(15.f, 15.f, .1f));

   
    d = min(d, opSmoothUnion(d, d7, 0.01));

    return vec2(d, 7.f);
}

vec2 mapObj1(vec3 pos)
{
    float d = 1e10;
    float d_1 = sdBox(pos + vec3(-7.5f, 2.5, -2.f), vec3(5.f, 5.f, 5.f));

    d = min(d, d_1);
    return vec2(d, 8.f);
}

vec2 mapObj2(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(7.5, 2.5, -2.f), vec3(5.f));

    d = min(d, d1);
    return vec2(d, 8.f);
}

vec2 mapObj3(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(7.5, -12.5f, -2.f), vec3(5.f));

    d = min(d, d1);
    return vec2(d, 8.f);
}

vec2 mapObj4(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(-7.5, -12.5f, -2.f), vec3(7.f, 7.f, 5.f));

    d = min(d, d1);
    return vec2(d, 8.f);
}





vec2 map(vec3 pos)
{

    float d = 1e10;
    float d2 = sdTorus(opTwist(pos + vec3(0.f, 0.f, -5.f)), vec2(0.3f, 0.03f));

    float tag = 0.f;

    float box_d1 = mapObj1(pos).x;
    float box_d2 = mapObj2(pos).x;
    float box_d3 = mapObj3(pos).x;
    float box_d4 = mapObj4(pos).x;

    if(box_d1 < 0.1)
    {
      float d1 = obj1(pos).x;
      if(d1 < d)
      {
        d = d1;
        tag = obj1(pos).y;
        box_d1 = d;
        //tag = 3.f;
      }
    }
    

    if(box_d2 < 0.1)
    {
      float d1 = obj2(pos).x;
      if(d1 < d)
      {
        d = d1;
        tag = obj2(pos).y;
        box_d2 = d;
        //tag = 3.f;
      }
    }

    if(box_d3 < 0.1)
    {
      float d1 = obj3(pos).x;
      float d2 = obj5(pos).x;
      if(d1 < d2 && d1 < d)
      {
        d = d1;
        tag = obj3(pos).y;
        box_d3 = d;
        //tag = 3.f;
      }
      else if(d2 < d1 && d2 < d)
      {
        d = d2;
        tag = obj5(pos).y;
        box_d3 = d;
      }
    }

    if(box_d4 < 0.1)
    {
      float d1 = obj4(pos).x;
      
      if(d1 < d)
      {
        d = d1;
        tag = obj4(pos).y;
        box_d4 = d;
        //tag = 3.f;
      }
    }

    // if(obj4(pos).x < d)
    //   {
    //     d = obj4(pos).x;
    //     tag = obj4(pos).y;
    //    // box_d1 = d;
    //     //tag = 3.f;
    //   }


    // if(obj3(pos).x < d)
    // {
    //     d = obj3(pos).x;
    //     tag = obj3(pos).y;
    //     //tag = 3.f;
    // }

    // if(obj1(pos).x < d)
    // {
    //     d = obj1(pos).x;
    //     tag = obj1(pos).y;
    //     ///tag = 1.f;
        
    // }

    // if(obj2(pos).x < d)
    // {
    //     d = obj2(pos).x;
    //     tag = obj2(pos).y;
    //     //tag = 2.f;
    // }

 

    // if(obj4(pos).x < d)
    // {

        
    //     d = obj4(pos).x;
    //     tag = obj4(pos).y;

    // }

    // if(obj5(pos).x < d)
    // {
    //     d = obj5(pos).x;
    //     tag = obj5(pos).y;
      
    // }

    // if(obj6(pos).x < d)
    // {
    //     d = obj6(pos).x;
    //     tag = obj6(pos).y;
   
    // }


    if(obj7(pos).x < d)
    {
        d = obj7(pos).x;
        tag = obj7(pos).y;
        //tag = 7.f;
    }

    if(obj8(pos).x < d)
    {
        d = obj8(pos).x;
        tag = obj8(pos).y;
    }

    d = min(d, min(box_d1, (min(box_d2, min(box_d3, box_d4)))));

    


    // if(mapObj1(pos).x < d)
    // {
    //   d = mapObj1(pos).x;
    //   tag = mapObj1(pos).y;
    // }

    // if(mapObj2(pos).x < d)
    // {
    //   d = mapObj2(pos).x;
    //   tag = mapObj2(pos).y;
    // }

    // if(mapObj3(pos).x < d)
    // {
    //   d = mapObj3(pos).x;
    //   tag = mapObj3(pos).y;
    // }

    // if(mapObj4(pos).x < d)
    // {
    //   d = mapObj4(pos).x;
    //   tag = mapObj4(pos).y;
    // }
    //float d2_mix = mix(d2_1, d2, cos(u_Time * 0.01));
    //d = min(d, d2);

    //d = min(d, opSmoothSubtraction(d1, d1_1, 0.5));

    // d = min(d, opSmoothUnion(d1, d1_1, 0.1));
    // d = min(d, opSmoothUnion(d1, d1_2, 0.1));
    // vec3 q = pos - vec3(1.0,0.0,1.0);
    //d = min(d, opSmoothSubtraction(d2, d3, 0.1));

     //d = min(d, box_d1);
    return vec2(d, tag);

}

vec3 normal(vec3 p) 
{
    float EPSILON = 0.1f;
    return normalize(vec3(
        map(vec3(p.x + EPSILON, p.y, p.z)).x - map(vec3(p.x - EPSILON, p.y, p.z)).x,
        map(vec3(p.x, p.y + EPSILON, p.z)).x - map(vec3(p.x, p.y - EPSILON, p.z)).x,
        map(vec3(p.x, p.y, p.z + EPSILON)).x - map(vec3(p.x, p.y, p.z - EPSILON)).x
    ));
}




vec2 dist(vec3 dir, vec3 eye)
{
  float depth = 0.001f;
  float tag;
for (int i = 0; i < 64; i++) {
    //float dist = sdHexPrism(eye + depth * dir, vec2(3.f, 3.f));
    //float dist = sdTorus(eye + depth * dir, vec2(3.f, 5.f));
    //float dist = sphere(eye + depth * dir);
    float dist = map(eye + depth * dir).x;
    tag = map(eye + depth * dir).y;
    if (dist <= 0.01f || depth > 100.f) {
        break;
    }
    // Move along the view ray
    depth += dist;
}
return vec2(depth, tag);
}

float opDisplace(vec3 p, vec3 eye )
{
    float d1 = dist(p, eye).x;
    float d2 = d1 * sin(20.f *p.x)*sin(20.f *p.y)*sin(20.f*p.z);
    return d1+d2;
}



mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
	vec3 f = normalize(center - eye);
	vec3 s = normalize(cross(f, up));
	vec3 u = cross(s, f);
	return mat4(
		vec4(s, 0.0),
		vec4(u, 0.0),
		vec4(-f, 0.0),
		vec4(0.0, 0.0, 0.0, 1)
	);
}



void main() {

  //vec2 screen = vec2((fs_Pos.x / u_Dimensions.x) * 2.f - 1.f, 1.f - (fs_Pos.y / u_Dimensions.y) * 2.f);

//get ray marching direction 
  vec2 screen = fs_Pos;
  vec3 right = cross(normalize(u_Ref - u_Eye), normalize(u_Up));
  float alpha = 50.f * 3.14159 / 180.0;
  float len = length(u_Ref - u_Eye);
  vec3 V = u_Up * len * tan(alpha);
  vec3 H = right * len * u_Dimensions.x / u_Dimensions.y * tan(alpha);
  vec3 P = u_Ref + screen.x * H + screen.y * V;
  vec3 dir = normalize(P - u_Eye);


//intersection
  float t = dist(dir, u_Eye).x;
  float tag = dist(dir, u_Eye).y;

  //float t = opDisplace(dir, u_Eye);
  vec3 isect = u_Eye + t * dir;

//phong (called normal function)
  vec3 V_P = normalize(u_Eye - P);
  vec3 L_P = normalize(vec3(-5.f) - P);
  vec3 H_P = (V_P + L_P) / 2.f;




  float spec = abs(pow(dot(H_P, normalize(normal(isect))), 5.f));

  float lambert = dot(normalize(-normal(isect)), vec3(-1.f));

//SDF
  vec3 color;
  if(t < 100.f)
  {
    //tag = 1.f;
    int tag_int = int(tag);
    switch(tag_int)
    {
      case 0: color = vec3(0.9137, 0.7176, 0.3569);break;
      case 1: color = vec3(0.4157, 0.4706, 0.9725);break;
      case 3: color = vec3(0.9059, 0.5137, 0.6);break;
      case 2: color = vec3(0.698, 0.5412, 0.9529);break;
      case 4: color = vec3(0.7451, 0.9412, 0.4902);break;
      case 7: vec3 color1 = vec3(0.3529, 0.2157, 0.0275);
              float height = mix(0.f, 1.f, isect.y);
              //float perlin = cnoise(vec3(height+ isect.y) * 0.3);
              float perlin = pnoise(vec3(isect), vec3(2.f));
              vec3 color2 = vec3(0.9059, 0.7216, 0.3765);
              color = mix(color2, color1, perlin);break;
      case 8: vec3 color3 = vec3(0.3529, 0.2157, 0.0275);
              float height1 = mix(0.f, 1.f, isect.z);
              float perlin1 = cnoise(vec3(height+ isect.z) * 0.3);
              vec3 color4 = vec3(0.9059, 0.7216, 0.3765);
              color = mix(color2, color1, perlin1);break;
    }

    //color = vec3(tag / 4.0);
     //color = vec3(0.8667, 0.5882, 0.0667);
     //color = pow( color, vec3(0.4545) );
     out_Col = vec4(color * (spec* 0.2 + lambert * 0.8), 1.f);
  }
  else
  {
    //  color = 0.5 * (vec3(1.0));
    //  out_Col = vec4(color, 1.f);

     float perlin = cnoise(isect * 0.1f * sin(u_Time * 0.01));

    // float scale = mix(0.5f, 1.f, 0.1 * perlin);
     out_Col = vec4(perlin * vec3(0.4549, 0.7098, 0.7529), 1.f);
    // out_Col = vec4(pow( vec3(out_Col), vec3(0.4545)), 1.f);
  }


}

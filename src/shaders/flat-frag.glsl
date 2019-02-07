#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 perm(vec4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noise(vec3 p){
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}
vec2 rand(vec2 p)
{
    return fract(sin(vec2(dot(p,vec2(127.1,311.7)),
                          dot(p, vec2(269.5, 183.3))))
                 * 43758.5453);
}
/*
vec4 noise(vec2 p)
{

    float distance = 1000.f;

    float width = 1.0f / u_Dimensions.x;
    float height = 1.0f / u_Dimensions.y;

    vec2 randPos;

    ivec2 cell_Pos = ivec2(int(p.x / width), int(p.y / height));

    vec3 col;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            vec2 tmpCell = vec2(clamp(1.0f * float((cell_Pos.x + i - 1)) * width, 0.f, 1.f),
                                clamp(1.0f * float((cell_Pos.y + j - 1)) * height, 0.f, 1.f));


            randPos = tmpCell + vec2(rand(tmpCell).x * width, rand(tmpCell).y * height) ;

            vec2 disVec = vec2((randPos - p).x / width, (randPos - p).y / height) ;

            if(distance > length(disVec))
            {
                distance = length(disVec);

                col = texture(u_RenderedTexture, randPos).xyz;
            }
        }
    }
    return vec4(col, distance);
}

*/


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

    pos = pos / 3.f;
    float d = 1e10;
    float d1 = sphere(pos, 1.f);
    //float noise = noise(pos);
    float d1_mix = mix(d1, d1 + 0.3 * sin(10.f * pos.x)*sin(8.f * pos.y)*sin(6.f * pos.z), cos(u_Time * 0.01));
    d = min(d, d1_mix);
    

    return vec2(d * 3.f, 1.f);
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

    vec3 displace = vec3(2.f * sin(u_Time * 3.14 * 0.01), 0.f, 2.f * cos(u_Time * 3.14 * 0.01));
    vec3 displaceR = vec3(2.f * sin(-u_Time * 3.14 * 0.01), 0.f, 2.f * cos(-u_Time * 3.14 * 0.01));
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

//heart
vec2 obj3(vec3 pos)
{
    float d = 1e10;
    pos = pos + vec3(7.5, -10.f, -2.0);
    float scale = impulse(1.f, abs(sin(u_Time * 3.14 * 0.05)));
    pos /= 2.f * scale;

    mat4 r = rotateZ(3.14 / 6.f);
    mat4 rr = rotateZ(-3.14 / 6.f);

    float d1 = sdRoundCone(vec3(r * vec4(pos, 1.f)), 0.1, 0.5 , 1.0);
    float d2 = sdRoundCone(vec3(rr * vec4(pos, 1.f)), 0.1, 0.5 , 1.0);

    d = min(d, opSmoothUnion(d1, d2, 0.1));

    return vec2(d * scale , 3.f);

}

//
vec2 obj4(vec3 pos)
{
    float d = 1e10;

    float noise = noise(pos);

    float r = mix(0.8, 1.2, abs(sin(u_Time * 0.01)));
    r = 1.21;
   //float  r = noise ;

    float d1 = sphere(pos + vec3(0.f , 0.f, -4.f ), 1.f);
    float d2_1 = sphere(pos + vec3(0.f , -1.5f , -4.f ), r);
    float d2_2 = sphere(pos + vec3(0.f , 0.f , -5.5f ), r);
    float d2_3 = sphere(pos + vec3(1.5f , 0.f , -4.f ), r);
    float d2_4 = sphere(pos + vec3(-1.5f , 0.f , -4.f ), r);
    float d2_5 = sphere(pos + vec3(0.f , 1.5f , -4.f ), r);
    float d2_6 = sphere(pos + vec3(0.f , 0.f , -2.5f ), r);
    // d = min(d, d1);5

    float d_union = opSmoothUnion(d2_1, d2_2, 0.1);
    d_union = opSmoothUnion(d_union, d2_3, 0.1);
    d_union = opSmoothUnion(d_union, d2_4, 0.1);
    d_union = opSmoothUnion(d_union, d2_5, 0.1);
    d_union = opSmoothUnion(d_union, d2_6, 0.1);
    //d = min(d, opSmoothSubtraction(d1, d2_1, 0.1));
    d = min(d, opSmoothSubtraction(d1, d_union, 0.1));

        float d_mix = d + 0.5f * noise;

        d_mix = mix(d, d_mix, sin(u_Time * 0.01));


        d = min(d, d_mix);
    
    // d = min(d, d2_2);

    // vec3 q = pos - vec3(0.0,0.0,1.0);
    // float d1 = sphere( q-vec3(0.0,0.5+0.3,0.0), 0.55 );
    // float d2 = sdRoundBox(q, vec3(0.6,0.2,0.7), 0.1 ); 
    // float dt = opSmoothSubtraction(d2,d1, 0.25);
    // d = min( d, dt );

    return vec2(d, 4.f);

}

//roundbox
vec2 obj5(vec3 pos)
{
    float d = 1e10;

    pos = pos + vec3(7.5, -11.f, -2.0);

    float scale = mix(2.f, 1.f, abs(sin(u_Time * 0.01)));
    pos /= 2.f;
    
    float d1 = sdRoundBox(pos, vec3(1.f), 0.05);
    
	  float d2 = sphere(pos, 1.4);

    float ds = min(d, opSmoothSubtraction(d2, d1, 0.1));
    d = min(d, ds);

    return vec2(d * 2.f, 3.f);

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
vec2 obj7(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(0.f, -5.f, 0.f), vec3(15.f, .1f, 2.f));
    float d2 = sdBox(pos + vec3(0.f, 10.f, 0.f), vec3(15.f, .1f, 2.f));
    float d3 = sdBox(pos + vec3(0.f, -20.f, 0.f), vec3(15.f, .1f, 2.f));
    float d4 = sdBox(pos + vec3(0.f, -5.f, 0.f), vec3(.1f, 15.f, 2.f));
    float d5 = sdBox(pos + vec3(-15.f, -5.f, 0.f), vec3(.1f, 15.f, 2.f));
    float d6 = sdBox(pos + vec3(15.f, -5.f, 0.f), vec3(.1f, 15.f, 2.f));


    d = min(d, d1);
    d = min(d, d2);
    d = min(d, d3);
    d = min(d, opSmoothUnion(d, d4, 0.01));
    d = min(d, opSmoothUnion(d, d5, 0.01));
    d = min(d, opSmoothUnion(d, d6, 0.01));

    return vec2(d, 7.f);
}

float mapObj1(vec3 pos)
{
    float d = 1e10;
    float d_1 = sdBox(pos + vec3(-7.5f, 2.5, -2.f), vec3(15.f, 15.f, 2.f));

    d = min(d, d_1);
    return d;
}

bool reachObj1(vec3 dir, vec3 eye)
{
  float depth = 0.001f;
  float tag;
for (int i = 0; i < 64; i++) {
    //float dist = sdHexPrism(eye + depth * dir, vec2(3.f, 3.f));
    //float dist = sdTorus(eye + depth * dir, vec2(3.f, 5.f));
    //float dist = sphere(eye + depth * dir);
    float dist = mapObj1(eye + depth * dir);
    tag = mapObj1(eye + depth * dir);
    if (dist <= 0.01f || depth > 100.f) {
        return true;
    }
    // Move along the view ray
    depth += dist;
}
return false;
}






vec2 map(vec3 pos)
{
    //pos += vec3(5.f, 0.f, 0.f);
    float d = 1e10;
    //float offset = smoothstep(0, 10.f*fs_Pos.y, cos(u_Time * 0.05));

    // pos += vec3(-3.f, 0.f, 0.f);

    // float d1 = sphere(pos + vec3(-3.f , 0.f, -4.f ), 1.f);
    // float d1_1 = sphere(pos + vec3(-7.f, 0.f, -4.f) + 0.1 * vec3(sin(pos.x), sin(pos.y), sin(pos.z)), 1.f);
    // float d1_2 = sphere(pos + vec3(-3.f, -3.5f, 0.f), 1.f);
    float d2 = sdTorus(opTwist(pos + vec3(0.f, 0.f, -5.f)), vec2(0.3f, 0.03f));

    float tag = 0.f;

    // if(reachObj1)
    // {
    //     d = obj1(pos).x;
    //     tag = obj1(pos).y;
    // }


    if(obj3(pos).x < d)
    {
        d = obj3(pos).x;
        tag = obj3(pos).y;
        //tag = 3.f;
    }

    if(obj1(pos).x < d)
    {
        d = obj1(pos).x;
        tag = obj1(pos).y;
        ///tag = 1.f;
        
    }

    if(obj2(pos).x < d)
    {
        d = obj2(pos).x;
        tag = obj2(pos).y;
        //tag = 2.f;
    }

 

    // if(obj4(pos).x < d)
    // {

        
    //     d = obj4(pos).x;
    //     tag = obj4(pos).y;

    // }

    if(obj5(pos).x < d)
    {
        d = obj5(pos).x;
        tag = obj5(pos).y;
      
    }

    if(obj6(pos).x < d)
    {
        d = obj6(pos).x;
        tag = obj6(pos).y;
   
    }


    if(obj7(pos).x < d)
    {
        d = obj7(pos).x;
        tag = obj7(pos).y;
        //tag = 7.f;
    }
    //float d2_mix = mix(d2_1, d2, cos(u_Time * 0.01));
    //d = min(d, d2);

    //d = min(d, opSmoothSubtraction(d1, d1_1, 0.5));

    // d = min(d, opSmoothUnion(d1, d1_1, 0.1));
    // d = min(d, opSmoothUnion(d1, d1_2, 0.1));
    // vec3 q = pos - vec3(1.0,0.0,1.0);
    //d = min(d, opSmoothSubtraction(d2, d3, 0.1));

    // d = min(d, d2_5);
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
      case 0: color = vec3(0.8667, 0.5882, 0.0667);break;
      case 1: color = vec3(0.1804, 0.251, 0.8784);break;
      case 2: color = vec3(0.9529, 0.3294, 0.4627);break;
      case 3: color = vec3(0.7765, 0.7882, 0.2314);break;
    }

    //color = vec3(tag / 4.0);
     //color = vec3(0.8667, 0.5882, 0.0667);
     //color = pow( color, vec3(0.4545) );
     out_Col = vec4(color * (spec* 0.2 + lambert * 0.8), 1.f);
  }
  else
  {
     color = 0.5 * (vec3(1.0));
     out_Col = vec4(color, 1.f);
  }


}

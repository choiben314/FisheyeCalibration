//System Includes
#include <cmath>
#include <limits>
#include <algorithm>
#include <bitset>
#include <iostream>

//Intrinsic Includes
#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#endif

//Module Includes
#include "FRF.h"

#define PI 3.14159265358979

//We start with standard and compound-type field encoders and decoders (serialisers and deserializers). The decoders all assume that size
//checking has been performed in advance and the needed bytes are present. The only exceptions are the decoders for MatX objects and Strings.
//These have variable sizes so size-checking needs to be performed in the decoder functions. They take a MaxBytes argument and make sure that
//no more than the provided number of bytes are used by the object being decoded.

// ****************************************************************************************************************************************
// ***************************************************   Standard-type Field encoders   ***************************************************
// ****************************************************************************************************************************************
static void encodeField_uint8 (std::vector<uint8_t> & Buffer, uint8_t  x) { Buffer.push_back(x); }

static void encodeField_uint16 (std::vector<uint8_t> & Buffer, uint16_t x) {
	Buffer.push_back((uint8_t) (x >> 8 ));
	Buffer.push_back((uint8_t)  x       );
}

static void encodeField_uint32 (std::vector<uint8_t> & Buffer, uint32_t x) {
	Buffer.push_back((uint8_t) (x >> 24));
	Buffer.push_back((uint8_t) (x >> 16));
	Buffer.push_back((uint8_t) (x >> 8 ));
	Buffer.push_back((uint8_t)  x       );
}

static void encodeField_uint64 (std::vector<uint8_t> & Buffer, uint64_t x) {
	Buffer.push_back((uint8_t) (x >> 56));
	Buffer.push_back((uint8_t) (x >> 48));
	Buffer.push_back((uint8_t) (x >> 40));
	Buffer.push_back((uint8_t) (x >> 32));
	Buffer.push_back((uint8_t) (x >> 24));
	Buffer.push_back((uint8_t) (x >> 16));
	Buffer.push_back((uint8_t) (x >> 8 ));
	Buffer.push_back((uint8_t)  x       );
}

static void encodeField_int8   (std::vector<uint8_t> & Buffer, int8_t  x) {encodeField_uint8 (Buffer, (uint8_t)  x);}
static void encodeField_int16  (std::vector<uint8_t> & Buffer, int16_t x) {encodeField_uint16(Buffer, (uint16_t) x);}
static void encodeField_int32  (std::vector<uint8_t> & Buffer, int32_t x) {encodeField_uint32(Buffer, (uint32_t) x);}
static void encodeField_int64  (std::vector<uint8_t> & Buffer, int64_t x) {encodeField_uint64(Buffer, (uint64_t) x);}
static void encodeField_float32(std::vector<uint8_t> & Buffer, float  x)  {encodeField_uint32(Buffer, reinterpret_cast<uint32_t &>(x));}
static void encodeField_float64(std::vector<uint8_t> & Buffer, double x)  {encodeField_uint64(Buffer, reinterpret_cast<uint64_t &>(x));}


// ****************************************************************************************************************************************
// ***************************************************   Compound-type Field encoders   ***************************************************
// ****************************************************************************************************************************************
static void encodeField_Vec2 (std::vector<uint8_t> & Buffer, Eigen::Vector2d x) {
	encodeField_float64(Buffer, x(0));
	encodeField_float64(Buffer, x(1));
}

static void encodeField_Vec3 (std::vector<uint8_t> & Buffer, Eigen::Vector3d x) {
	encodeField_float64(Buffer, x(0));
	encodeField_float64(Buffer, x(1));
	encodeField_float64(Buffer, x(2));
}

static void encodeField_Vec4 (std::vector<uint8_t> & Buffer, Eigen::Vector4d x) {
	encodeField_float64(Buffer, x(0));
	encodeField_float64(Buffer, x(1));
	encodeField_float64(Buffer, x(2));
	encodeField_float64(Buffer, x(3));
}

static void encodeField_Mat2 (std::vector<uint8_t> & Buffer, Eigen::Matrix2d A) {
	encodeField_float64(Buffer, A(0,0));  encodeField_float64(Buffer, A(0,1));
	encodeField_float64(Buffer, A(1,0));  encodeField_float64(Buffer, A(1,1));
}

static void encodeField_Mat3 (std::vector<uint8_t> & Buffer, Eigen::Matrix3d A) {
	encodeField_float64(Buffer, A(0,0));  encodeField_float64(Buffer, A(0,1));  encodeField_float64(Buffer, A(0,2));
	encodeField_float64(Buffer, A(1,0));  encodeField_float64(Buffer, A(1,1));  encodeField_float64(Buffer, A(1,2));
	encodeField_float64(Buffer, A(2,0));  encodeField_float64(Buffer, A(2,1));  encodeField_float64(Buffer, A(2,2));
}

static void encodeField_Mat4 (std::vector<uint8_t> & Buffer, Eigen::Matrix4d A) {
	encodeField_float64(Buffer, A(0,0));  encodeField_float64(Buffer, A(0,1));  encodeField_float64(Buffer, A(0,2));  encodeField_float64(Buffer, A(0,3));
	encodeField_float64(Buffer, A(1,0));  encodeField_float64(Buffer, A(1,1));  encodeField_float64(Buffer, A(1,2));  encodeField_float64(Buffer, A(1,3));
	encodeField_float64(Buffer, A(2,0));  encodeField_float64(Buffer, A(2,1));  encodeField_float64(Buffer, A(2,2));  encodeField_float64(Buffer, A(2,3));
	encodeField_float64(Buffer, A(3,0));  encodeField_float64(Buffer, A(3,1));  encodeField_float64(Buffer, A(3,2));  encodeField_float64(Buffer, A(3,3));
}

static void encodeField_MatX (std::vector<uint8_t> & Buffer, Eigen::MatrixXd A) {
	encodeField_uint32(Buffer, (uint32_t) A.rows());
	encodeField_uint32(Buffer, (uint32_t) A.cols());
	for (int row = 0; row < A.rows(); row++) {
		for (int col = 0; col < A.cols(); col++)
			encodeField_float64(Buffer, A(row,col));
	}
}

static void encodeField_String (std::vector<uint8_t> & Buffer, const std::string & x) {
	encodeField_uint32(Buffer, (uint32_t) x.size());
	for (char item : x)
		Buffer.push_back((uint8_t) item);
}

static void encodeField_GPST (std::vector<uint8_t> & Buffer, uint32_t Week, double TOW) {
	encodeField_uint32(Buffer, Week);
	encodeField_float64(Buffer, TOW);
}


// ****************************************************************************************************************************************
// ***************************************************   Standard-type Field decoders   ***************************************************
// ****************************************************************************************************************************************
static uint8_t decodeField_uint8 (std::vector<uint8_t>::const_iterator & Iter) { return(*Iter++); }

static uint16_t decodeField_uint16 (std::vector<uint8_t>::const_iterator & Iter) {
	uint16_t value;
	value  = (uint16_t) *Iter++;  value <<= 8;
	value += (uint16_t) *Iter++;
	return(value);
}

static uint32_t decodeField_uint32 (std::vector<uint8_t>::const_iterator & Iter) {
	uint32_t value;
     value  = (uint32_t) *Iter++;  value <<= 8;
     value += (uint32_t) *Iter++;  value <<= 8;
     value += (uint32_t) *Iter++;  value <<= 8;
     value += (uint32_t) *Iter++;
     return(value);
}

static uint64_t decodeField_uint64 (std::vector<uint8_t>::const_iterator & Iter) {
	uint64_t value;
     value  = (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;  value <<= 8;
     value += (uint64_t) *Iter++;
     return(value);
}

static int8_t   decodeField_int8   (std::vector<uint8_t>::const_iterator & Iter) {return((int8_t) *Iter++);}
static int16_t  decodeField_int16  (std::vector<uint8_t>::const_iterator & Iter) {return((int16_t) decodeField_uint16(Iter));}
static int32_t  decodeField_int32  (std::vector<uint8_t>::const_iterator & Iter) {return((int32_t) decodeField_uint32(Iter));}
static int64_t  decodeField_int64  (std::vector<uint8_t>::const_iterator & Iter) {return((int64_t) decodeField_uint64(Iter));}

static float decodeField_float32 (std::vector<uint8_t>::const_iterator & Iter) {
	uint32_t bitPattern = decodeField_uint32(Iter);
	return reinterpret_cast<float &>(bitPattern);
}

static double decodeField_float64 (std::vector<uint8_t>::const_iterator & Iter) {
	uint64_t bitPattern = decodeField_uint64(Iter);
	return reinterpret_cast<double &>(bitPattern);
}


// ****************************************************************************************************************************************
// ***************************************************   Compound-type Field decoders   ***************************************************
// ****************************************************************************************************************************************
static Eigen::Vector2d decodeField_Vec2 (std::vector<uint8_t>::const_iterator & Iter) {
	Eigen::Vector2d v;
	v(0) = decodeField_float64(Iter);
	v(1) = decodeField_float64(Iter);
	return v;
}

static Eigen::Vector3d decodeField_Vec3 (std::vector<uint8_t>::const_iterator & Iter) {
	Eigen::Vector3d v;
	v(0) = decodeField_float64(Iter);
	v(1) = decodeField_float64(Iter);
	v(2) = decodeField_float64(Iter);
	return v;
}

static Eigen::Vector4d decodeField_Vec4 (std::vector<uint8_t>::const_iterator & Iter) {
	Eigen::Vector4d v;
	v(0) = decodeField_float64(Iter);
	v(1) = decodeField_float64(Iter);
	v(2) = decodeField_float64(Iter);
	v(3) = decodeField_float64(Iter);
	return v;
}

static Eigen::Matrix2d decodeField_Mat2 (std::vector<uint8_t>::const_iterator & Iter) {
	Eigen::Matrix2d A;
	A(0,0) = decodeField_float64(Iter);  A(0,1) = decodeField_float64(Iter);
	A(1,0) = decodeField_float64(Iter);  A(1,1) = decodeField_float64(Iter);
	return A;
}

static Eigen::Matrix3d decodeField_Mat3 (std::vector<uint8_t>::const_iterator & Iter) {
	Eigen::Matrix3d A;
	A(0,0) = decodeField_float64(Iter);  A(0,1) = decodeField_float64(Iter);  A(0,2) = decodeField_float64(Iter);
	A(1,0) = decodeField_float64(Iter);  A(1,1) = decodeField_float64(Iter);  A(1,2) = decodeField_float64(Iter);
	A(2,0) = decodeField_float64(Iter);  A(2,1) = decodeField_float64(Iter);  A(2,2) = decodeField_float64(Iter);
	return A;
}

static Eigen::Matrix4d decodeField_Mat4 (std::vector<uint8_t>::const_iterator & Iter) {
	Eigen::Matrix4d A;
	A(0,0) = decodeField_float64(Iter);  A(0,1) = decodeField_float64(Iter);  A(0,2) = decodeField_float64(Iter);  A(0,3) = decodeField_float64(Iter);
	A(1,0) = decodeField_float64(Iter);  A(1,1) = decodeField_float64(Iter);  A(1,2) = decodeField_float64(Iter);  A(1,3) = decodeField_float64(Iter);
	A(2,0) = decodeField_float64(Iter);  A(2,1) = decodeField_float64(Iter);  A(2,2) = decodeField_float64(Iter);  A(2,3) = decodeField_float64(Iter);
	A(3,0) = decodeField_float64(Iter);  A(3,1) = decodeField_float64(Iter);  A(3,2) = decodeField_float64(Iter);  A(3,3) = decodeField_float64(Iter);
	return A;
}

//MaxBytes is the maximum number of bytes that can belong to the full Matrix object. This is a reference argument that will be decremented by the number of decoded bytes.
//If the dimensions of the matrix are such that the object would exceed this number of bytes, then an empty matrix is returned and the interator is advanced by MaxBytes.
static Eigen::MatrixXd decodeField_MatX (std::vector<uint8_t>::const_iterator & Iter, unsigned int & MaxBytes) {
	if (MaxBytes < 8U) {
		fprintf(stderr,"Warning in decodeField_MatX: Not enough bytes left for matrix. Aborting matrix decode.\r\n");
		Iter += MaxBytes;
		MaxBytes = 0U;
		return Eigen::MatrixXd(0,0);
	}
	
	uint32_t rows = decodeField_uint32(Iter);
	uint32_t cols = decodeField_uint32(Iter);
	unsigned int bytesForObject = rows*cols*8U + 8U;
	if (bytesForObject > MaxBytes) {
		fprintf(stderr,"Warning in decodeField_MatX: Not enough bytes left for matrix. Aborting matrix decode.\r\n");
		Iter += MaxBytes;
		MaxBytes = 0U;
		return Eigen::MatrixXd(0,0);
	}
	
	Eigen::MatrixXd A(rows, cols);
	for (int row = 0; row < (int) rows; row++) {
		for (int col = 0; col < (int) cols; col++)
			A(row,col) = decodeField_float64(Iter);
	}
	MaxBytes -= bytesForObject;
	return A;
}

//MaxBytes is the maximum number of bytes that can belong to the full string object (size field included). This is a reference argument that will be decremented by the number of decoded bytes.
//If the string field advertises a length that would make the full string object exceed this number of bytes, then we replace the advertised length with the largest safe value.
static std::string decodeField_String (std::vector<uint8_t>::const_iterator & Iter, unsigned int & MaxBytes) {
	if (MaxBytes < 4U) {
		fprintf(stderr,"Warning in decodeField_String: Not enough bytes left for string. Aborting decode.\r\n");
		Iter += MaxBytes;
		MaxBytes = 0U;
		return std::string();
	}
	
	unsigned int size = (unsigned int) decodeField_uint32(Iter);
	unsigned int bytesForObject = size + 4U;
	if (bytesForObject > MaxBytes) {
		fprintf(stderr,"Warning in decodeField_String: Not enough bytes left for string. Aborting decode.\r\n");
		Iter += MaxBytes;
		MaxBytes = 0U;
		return std::string();
	}
	
	std::string S;
	for (unsigned int n = 0U; n < size; n++)
		S.push_back((char) *Iter++);
	MaxBytes -= bytesForObject;
	return(S);
}

static std::tuple<uint32_t,double> decodeField_GPST (std::vector<uint8_t>::const_iterator & Iter) {
	uint32_t Week = decodeField_uint32(Iter);
	double   TOW  = decodeField_float64(Iter);
	return std::make_tuple(Week, TOW);
}


// ****************************************************************************************************************************************
// *********************************************   High-level Serializers and Deserializers   *********************************************
// ****************************************************************************************************************************************
static void serialize_LayerManifestSection(const FRFLayer * Layer, std::vector<uint8_t> & Buffer) {
	encodeField_String (Buffer, Layer->Name);
	encodeField_String (Buffer, Layer->Description);
	encodeField_int32  (Buffer, Layer->UnitsCode);
	encodeField_uint8  (Buffer, Layer->GetTypeCode());
	encodeField_float64(Buffer, Layer->alpha);
	encodeField_float64(Buffer, Layer->beta);
	encodeField_uint8  (Buffer, Layer->HasValidityMask ? (uint8_t) 1U : (uint8_t) 0U);
}

//Populate all fields but Data in an FRFLayer. MaxBytes is the largest size the section is allowed to be, and it should be verified in advance that
//at least this many bytes are available in the buffer. If, in the course of deserialization, it looks like the section uses more than this many bytes,
//it will result in an error. This is a reference argument and it will be decremented by the number of bytes decoded.
static void deserialize_LayerManifestSection(FRFLayer * Layer, std::vector<uint8_t>::const_iterator & BufferIter, unsigned int & MaxBytes) {
	Layer->Name        = decodeField_String(BufferIter, MaxBytes);
	Layer->Description = decodeField_String(BufferIter, MaxBytes);
	
	if (MaxBytes < 22U) {
		fprintf(stderr, "Error in deserialize_LayerManifestSection. Not enough bytes left for section decoding.\r\n");
		BufferIter += MaxBytes;
		MaxBytes = 0U;
		return;
	}
	
	Layer->UnitsCode       = decodeField_int32(BufferIter);
	Layer->SetTypeCode(      decodeField_uint8(BufferIter) );
	Layer->alpha           = decodeField_float64(BufferIter);
	Layer->beta            = decodeField_float64(BufferIter);
	uint8_t MaskFlag       = decodeField_uint8(BufferIter);
	Layer->HasValidityMask = (MaskFlag > 0U) ? true : false;
	
	MaxBytes -= 22U;
}

static void serialize_VisualizationSection(const FRFVisualization * Visualization, std::vector<uint8_t> & Buffer) {
	encodeField_String (Buffer, Visualization->Name);
	encodeField_String (Buffer, Visualization->Description);
	encodeField_uint32 (Buffer, Visualization->isRGB() ? (uint32_t) 0U : (uint32_t) 1U); //Encode VisCode
	if (Visualization->isRGB()) {
		const FRFVisualizationRGB * RGBVis = dynamic_cast <const FRFVisualizationRGB *> (Visualization);
		
		encodeField_uint32 (Buffer, 54U); //An RGB visualization payload has 54 bytes in it
		encodeField_uint16 (Buffer, RGBVis->RedIndex);
		encodeField_float64(Buffer, RGBVis->RedMin);
		encodeField_float64(Buffer, RGBVis->RedMax);
		encodeField_uint16 (Buffer, RGBVis->GreenIndex);
		encodeField_float64(Buffer, RGBVis->GreenMin);
		encodeField_float64(Buffer, RGBVis->GreenMax);
		encodeField_uint16 (Buffer, RGBVis->BlueIndex);
		encodeField_float64(Buffer, RGBVis->BlueMin);
		encodeField_float64(Buffer, RGBVis->BlueMax);
	}
	else if (Visualization->isColormap()) {
		const FRFVisualizationColormap * ColormapVis = dynamic_cast <const FRFVisualizationColormap *> (Visualization);
		
		encodeField_uint32 (Buffer, ColormapVis->SetPoints.size()*32U + 2U); //Number of bytes in a colormap visualization payload
		encodeField_uint16 (Buffer, ColormapVis->LayerIndex);
		for (std::tuple<double,double,double,double> const & setPoint : ColormapVis->SetPoints) {
			encodeField_float64(Buffer, std::get<0>(setPoint));
			encodeField_float64(Buffer, std::get<1>(setPoint));
			encodeField_float64(Buffer, std::get<2>(setPoint));
			encodeField_float64(Buffer, std::get<3>(setPoint));
		}
	}
	else
		fprintf(stderr,"Error in serialize_VisualizationSection: Unrecognized visualization type.\r\n");
}

//Returns ptr to new object, owned by the caller. MaxBytes is the largest size the section is allowed to be, and it should be verified in advance that at least
//this many bytes are available in the buffer. If, in the course of deserialization, it looks like the section uses more than this many bytes, it will result in an error.
//This is a reference argument and it will be decremented by the number of bytes decoded.
static FRFVisualization * deserialize_VisualizationSection(std::vector<uint8_t>::const_iterator & BufferIter, unsigned int & MaxBytes) {
	std::string Name        = decodeField_String(BufferIter, MaxBytes);
	std::string Description = decodeField_String(BufferIter, MaxBytes);
	
	if (MaxBytes < 8U) {
		fprintf(stderr, "Error in deserialize_VisualizationSection. Not enough bytes left for section decoding.\r\n");
		BufferIter += MaxBytes;
		MaxBytes = 0U;
		return NULL;
	}
	
	uint32_t VisCode = decodeField_uint32(BufferIter);
	uint32_t VisPayloadSize = decodeField_uint32(BufferIter);
	MaxBytes -= 8U;
	
	if (VisCode == 0U) {
		//This is an RGB visualization
		if ((VisPayloadSize != 54U) || (VisPayloadSize > MaxBytes)) {
			fprintf(stderr, "Error in deserialize_VisualizationSection. Not enough bytes left or incorrect payload size.\r\n");
			BufferIter += MaxBytes;
			MaxBytes = 0U;
			return NULL;
		}
		
		FRFVisualizationRGB * RGBVis = new FRFVisualizationRGB();
		RGBVis->Name        = Name;
		RGBVis->Description = Description;
		RGBVis->RedIndex    = decodeField_uint16 (BufferIter);
		RGBVis->RedMin      = decodeField_float64(BufferIter);
		RGBVis->RedMax      = decodeField_float64(BufferIter);
		RGBVis->GreenIndex  = decodeField_uint16 (BufferIter);
		RGBVis->GreenMin    = decodeField_float64(BufferIter);
		RGBVis->GreenMax    = decodeField_float64(BufferIter);
		RGBVis->BlueIndex   = decodeField_uint16 (BufferIter);
		RGBVis->BlueMin     = decodeField_float64(BufferIter);
		RGBVis->BlueMax     = decodeField_float64(BufferIter);
		
		MaxBytes -= VisPayloadSize;
		return dynamic_cast <FRFVisualization *> (RGBVis);
	}
	else if (VisCode == 1U) {
		//This is a Colormap visualization
		if ((VisPayloadSize < 34U) || (VisPayloadSize % 32U != 2U) || (VisPayloadSize > MaxBytes)) {
			fprintf(stderr, "Error in deserialize_VisualizationSection. Not enough bytes left or incorrect payload size.\r\n");
			BufferIter += MaxBytes;
			MaxBytes = 0U;
			return NULL;
		}
		
		FRFVisualizationColormap * ColormapVis = new FRFVisualizationColormap();
		ColormapVis->Name        = Name;
		ColormapVis->Description = Description;
		ColormapVis->LayerIndex  = decodeField_uint16 (BufferIter);
		unsigned int numberOfSetPoints = (VisPayloadSize - 2U) / 32U;
		for (unsigned int n = 0U; n < numberOfSetPoints; n++) {
			std::tuple<double,double,double,double> setPoint;
			std::get<0>(setPoint) = decodeField_float64(BufferIter);
			std::get<1>(setPoint) = decodeField_float64(BufferIter);
			std::get<2>(setPoint) = decodeField_float64(BufferIter);
			std::get<3>(setPoint) = decodeField_float64(BufferIter);
			ColormapVis->SetPoints.push_back(setPoint);
		}
		
		//Sort the set points by increasing order of value
		std::sort(ColormapVis->SetPoints.begin(), ColormapVis->SetPoints.end());
		
		MaxBytes -= VisPayloadSize;
		return dynamic_cast <FRFVisualization *> (ColormapVis);
	}
	else {
		//The visualization type is not recognized. We will not be able to properly interpret the data, but we can at least
		//strip off the right number of bytes so that we can continue deserialization.
		fprintf(stderr,"Warning: Unrecognized visualization type. Droping it.\r\n");
		if (VisPayloadSize > MaxBytes) {
			fprintf(stderr, "Error in deserialize_VisualizationSection. Not enough bytes left or incorrect payload size.\r\n");
			BufferIter += MaxBytes;
			MaxBytes = 0U;
			return NULL;
		}
		else {
			BufferIter += VisPayloadSize;
			MaxBytes -= VisPayloadSize;
			return NULL;
		}
	}
}

//Serialize a GeoTagging block payload
static void serialize_GeoTagBlockPayload(const FRFGeoTag * GeoTag, std::vector<uint8_t> & Buffer) {
	encodeField_Vec3 (Buffer, GeoTag->P_ECEF);
	encodeField_Mat3 (Buffer, GeoTag->C_ECEF_Cam);
	encodeField_GPST (Buffer, GeoTag->GPST_Week, GeoTag->GPST_TOW);
}

//Deserialize a GeoTagging block payload - returns ptr to new object, owned by the caller. No size checking is performed internally
static FRFGeoTag * deserialize_GeoTagPayload(std::vector<uint8_t>::const_iterator & BufferIter) {
	FRFGeoTag * GeoTag = new FRFGeoTag();
	GeoTag->P_ECEF     = decodeField_Vec3 (BufferIter);
	GeoTag->C_ECEF_Cam = decodeField_Mat3 (BufferIter);
	std::tuple<uint32_t,double> GPST = decodeField_GPST (BufferIter);
	GeoTag->GPST_Week = std::get<0>(GPST);
	GeoTag->GPST_TOW  = std::get<1>(GPST);
	return GeoTag;
}

//Serialize a GeoRegistration block payload
static void serialize_GeoRegistrationPayload(const FRFGeoRegistration * GeoRegistration, std::vector<uint8_t> & Buffer) {
	encodeField_uint16  (Buffer, GeoRegistration->RegistrationType);
	encodeField_float64 (Buffer, GeoRegistration->Altitude);
	encodeField_uint16  (Buffer, GeoRegistration->GridWidthDivisor);
	encodeField_uint16  (Buffer, GeoRegistration->GridHeightDivisor);
	for (Eigen::Vector2d const & gridPoint : GeoRegistration->GridPointLatLons) {
		encodeField_float64 (Buffer, gridPoint(0));
		encodeField_float64 (Buffer, gridPoint(1));
	}
}

//Deserialize a GeoRegistration block payload - returns ptr to new object, owned by the caller. MaxBytes holds an upper bound on the number of
//bytes that this payload can occupy. It is a reference argument and will be decremented by the number of decoded bytes.
static FRFGeoRegistration * deserialize_GeoRegistrationPayload(std::vector<uint8_t>::const_iterator & BufferIter, unsigned int & MaxBytes) {
	if (MaxBytes < 14U) {
		fprintf(stderr, "Error in deserialize_GeoRegistrationPayload. Not enough bytes left.\r\n");
		BufferIter += MaxBytes;
		MaxBytes = 0U;
		return NULL;
	}
	
	FRFGeoRegistration * GeoRegistration = new FRFGeoRegistration();
	GeoRegistration->RegistrationType  = decodeField_uint16  (BufferIter);
	GeoRegistration->Altitude          = decodeField_float64 (BufferIter);
	GeoRegistration->GridWidthDivisor  = decodeField_uint16  (BufferIter);
	GeoRegistration->GridHeightDivisor = decodeField_uint16  (BufferIter);
	MaxBytes -= 14U;
	
	uint32_t numberOfGridPoints = (GeoRegistration->GridWidthDivisor + 1U)*(GeoRegistration->GridHeightDivisor + 1U);
	
	if (numberOfGridPoints*16U > MaxBytes) {
		fprintf(stderr, "Error in deserialize_GeoRegistrationPayload. Not enough bytes left for all grid points.\r\n");
		BufferIter += MaxBytes;
		MaxBytes = 0U;
		delete GeoRegistration;
		return NULL;
	}
	
	for (uint32_t n = 0U; n < numberOfGridPoints; n++) {
		double Lat = decodeField_float64 (BufferIter);
		double Lon = decodeField_float64 (BufferIter);
		GeoRegistration->GridPointLatLons.push_back(Eigen::Vector2d(Lat, Lon));
	}
	MaxBytes -= numberOfGridPoints*16U;
	return GeoRegistration;
}

//Serialize a Custom block payload
static void serialize_CustomBlockPayload(const FRFCustomBlock & CustomBlock, std::vector<uint8_t> & Buffer) {
	encodeField_uint64  (Buffer, CustomBlock.CustomBlockCode);
	for (uint8_t byte : CustomBlock.BlockPayload)
		Buffer.push_back(byte);
}


// ****************************************************************************************************************************************
// ******************************************   Type Conversion and number processing Helpers   *******************************************
// ****************************************************************************************************************************************
static uint64_t saturateRound_uint(double value, uint8_t NumberOfBits) {
	uint64_t bitPattern = 0U;
	if (value <= 0.0)
		return 0U;
	else if (value >= (double) std::numeric_limits<uint64_t>::max() - 0.5)
		bitPattern = std::numeric_limits<uint64_t>::max();
	else
		bitPattern = (uint64_t) (value + 0.5);
	
	//Saturate to the correct number of bits
	if (NumberOfBits == 0U)
		return 0U;
	else if (NumberOfBits == 64U)
		return bitPattern;
	else if (bitPattern >= ((uint64_t) 1U) << NumberOfBits)
		return (((uint64_t) 1U) << NumberOfBits) - 1U;
	else
		return bitPattern;
}

static int8_t saturateRound_int8(double value) {
	if (value <= (double) std::numeric_limits<int8_t>::lowest())
		return std::numeric_limits<int8_t>::lowest();
	else if (value >= (double) std::numeric_limits<int8_t>::max())
		return std::numeric_limits<int8_t>::max();
	else
		return ((int8_t) std::lround(value));
}

static int16_t saturateRound_int16(double value) {
	if (value <= (double) std::numeric_limits<int16_t>::lowest())
		return std::numeric_limits<int16_t>::lowest();
	else if (value >= (double) std::numeric_limits<int16_t>::max())
		return std::numeric_limits<int16_t>::max();
	else
		return ((int16_t) std::lround(value));
}

static int32_t saturateRound_int32(double value) {
	if (value <= (double) std::numeric_limits<int32_t>::lowest())
		return std::numeric_limits<int32_t>::lowest();
	else if (value >= (double) std::numeric_limits<int32_t>::max())
		return std::numeric_limits<int32_t>::max();
	else
		return ((int32_t) std::lround(value));
}

static int64_t saturateRound_int64(double value) {
	if (value <= (double) std::numeric_limits<int64_t>::lowest())
		return std::numeric_limits<int64_t>::lowest();
	else if (value >= (double) std::numeric_limits<int64_t>::max())
		return std::numeric_limits<int64_t>::max();
	else
		return ((int64_t) std::llround(value));
}

static double saturateToRange(double Value, double MinValue, double MaxValue) { return std::max(std::min(Value, MaxValue), MinValue); }

//View Buffer as a large bit array (bytes ordered by index, each byte coming Msb first) and store a bit pattern in the given bits.
//The bits that will be used will be the least significant bits of BitPattern. LastBit must not be less than FirstBit.
static void packBits(std::vector<uint8_t> & Buffer, uint64_t FirstBit, uint64_t LastBit, uint64_t BitPattern) {
	uint64_t firstByte = FirstBit/8U;
	
	//If we are compiling with BMI2 support and the bit pattern is 56 bits or less, we can use a BMI2-optimised implementation
	#if (defined(__GNUC__) && defined(__BMI2__)) || (defined(_MSC_VER) && defined(__AVX2__))
	if ((LastBit - FirstBit  < 56U) && (firstByte + 7U < Buffer.size())) {
		//The BitPattern is 7 bytes or less and so will fit in 8 bytes, regardless of alignment. We are also not up against the end of the buffer.
		uint64_t * ptr = (uint64_t *) &(Buffer[firstByte]); //Get pointer to 8 bytes holding current data
		uint64_t oldPattern = (uint64_t) _bswap64((int64_t) *ptr); //Read current data and correct Endianness
		
		//Build insertion mask
		uint64_t numberOfBits = LastBit - FirstBit + 1U;
		uint64_t mask = (((uint64_t) 1U) << numberOfBits) - 1U;
		mask = mask << ((64U - numberOfBits) - (FirstBit % 8U));
		
		//Combine current pattern and new data. Then store
		uint64_t newPattern = (oldPattern & ~mask) + _pdep_u64(BitPattern, mask);
		*ptr = (uint64_t) _bswap64((int64_t) newPattern);
		
		//std::cerr << "Old Pattern:     " << std::bitset<64>(oldPattern) << "\r\n";
		//std::cerr << "Mask       :     " << std::bitset<64>(mask) << "\r\n";
		//std::cerr << "Bit Pattern:     " << std::bitset<64>(BitPattern) << "\r\n";
		//std::cerr << "Shifted Pattern: " << std::bitset<64>(_pdep_u64(BitPattern, mask)) << "\r\n";
		//std::cerr << "New Pattern:     " << std::bitset<64>(newPattern) << "\r\n";
		return;
	}
	#endif
	
	//Generic implementation that does not rely on intrinsics
	uint64_t lastByte = LastBit/8U;
	
	//Write the bit pattern to the data buffer
	for (uint64_t ByteNum = firstByte; ByteNum <= lastByte; ByteNum++) {
		uint64_t patternByteNumber = lastByte - ByteNum;
		
		uint8_t myByte; //Grab our byte (aligned with target buffer), with 0's in unused locations
		if (patternByteNumber > 0U)
			myByte = (uint8_t) (BitPattern >> (8U*patternByteNumber - (7U - (LastBit % 8U))));
		else
			myByte = (uint8_t) (BitPattern << (7U - (LastBit % 8U)));
		
		//We need to write our byte, but preserve the existing values in bits that don't belong to us.
		//We do this with a read-modify-write
		uint8_t readMask = 0U;
		if (ByteNum == firstByte) {
			switch (FirstBit % 8U) {
				case 0U: readMask += 0b00000000; break;
				case 1U: readMask += 0b10000000; break;
				case 2U: readMask += 0b11000000; break;
				case 3U: readMask += 0b11100000; break;
				case 4U: readMask += 0b11110000; break;
				case 5U: readMask += 0b11111000; break;
				case 6U: readMask += 0b11111100; break;
				case 7U: readMask += 0b11111110; break;
			}
		}
		if (ByteNum == lastByte) {
			switch (LastBit % 8U) {
				case 0U: readMask += 0b01111111; break;
				case 1U: readMask += 0b00111111; break;
				case 2U: readMask += 0b00011111; break;
				case 3U: readMask += 0b00001111; break;
				case 4U: readMask += 0b00000111; break;
				case 5U: readMask += 0b00000011; break;
				case 6U: readMask += 0b00000001; break;
				case 7U: readMask += 0b00000000; break;
			}
		}
		Buffer[ByteNum] = (Buffer[ByteNum] & readMask) + myByte;
	}
}

//View Buffer as a large bit array (bytes ordered by index, each byte coming Msb first) and extract some of the bits. They will
//become the least significant bits of the return value. Unused bits in the return value are cleared. LastBit must not be less than FirstBit.
static uint64_t grabBits(const std::vector<uint8_t> & Buffer, uint64_t FirstBit, uint64_t LastBit) {
	uint64_t firstByte = FirstBit/8U;
	
	//If we are compiling with BMI2 support and the bit pattern is 56 bits or less, we can use a BMI2-optimised implementation
	#if (defined(__GNUC__) && defined(__BMI2__)) || (defined(_MSC_VER) && defined(__AVX2__))
	if ((LastBit - FirstBit  < 56U) && (firstByte + 7U < Buffer.size())) {
		//The BitPattern is 7 bytes or less and so will fit in 8 bytes, regardless of alignment. We are also not up against the end of the buffer.
		uint64_t * ptr = (uint64_t *) &(Buffer[firstByte]); //Get pointer to 8 bytes of data
		uint64_t rawBytes = (uint64_t) _bswap64((int64_t) *ptr); //Read data and correct Endianness
		
		//Build extraction mask
		uint64_t numberOfBits = LastBit - FirstBit + 1U;
		uint64_t mask = (((uint64_t) 1U) << numberOfBits) - 1U;
		mask = mask << ((64U - numberOfBits) - (FirstBit % 8U));
		
		//Extract only the bits we need
		return _pext_u64(rawBytes, mask);
	}
	#endif
	
	//Generic implementation that does not rely on any intrinsics
	uint64_t lastByte = LastBit/8U;
	uint64_t bitPattern = 0U;
	
	for (uint64_t ByteNum = firstByte; ByteNum <= lastByte; ByteNum++) {
		uint8_t byte = Buffer[ByteNum];
		
		//If this is the first byte, mask off unrelated bits
		if ((ByteNum == firstByte) && (FirstBit % 8U != 0U)) {
			switch (FirstBit % 8U) {
				case 1U: byte = byte & 0b01111111; break;
				case 2U: byte = byte & 0b00111111; break;
				case 3U: byte = byte & 0b00011111; break;
				case 4U: byte = byte & 0b00001111; break;
				case 5U: byte = byte & 0b00000111; break;
				case 6U: byte = byte & 0b00000011; break;
				case 7U: byte = byte & 0b00000001; break;
			}
		}
		
		//Add this byte to our bit pattern
		if (ByteNum == lastByte)
			bitPattern += ((uint64_t) byte) >> (7U - (LastBit % 8U));
		else
			bitPattern += ((uint64_t) byte) << ((lastByte - ByteNum)*8U - (7U - (LastBit % 8U)));
	}
	return bitPattern;
}


// ****************************************************************************************************************************************
// ****************************************************   Geo-Registration Helpers   ******************************************************
// ****************************************************************************************************************************************
//Add or subtract multiples of (rangeMax-rangeMin) to value to return a value in the range [rangeMin, rangeMax)
//This is useful for getting the principal range of an angle. For instance, to get the value of theta within [0, 2*pi) use:
//getPrincipalValue(theta, 0.0, 2*PI);
//This function was verified through extensive testing on 8-13-2014
static double getPrincipalValue(double value, double rangeMin, double rangeMax) {
	double val = fmod(value - rangeMin, rangeMax - rangeMin);
	if (val < 0.0) val += (rangeMax - rangeMin);
	return (val + rangeMin);
}

//Take a (Lat, Lon) pair and adjust Lon to make sure it is in the range [0, 2*Pi)
static Eigen::Vector2d MapLatLonToPrincipalRange(Eigen::Vector2d LatLon) {
	LatLon(1) = getPrincipalValue(LatLon(1), 0.0, 2.0*PI);
	return LatLon;
}

//Add or subtract 2*PI to Angle if doing so gets us closer to the reference angle
//This only checks adding or subtracting a single multiple of 2*Pi, so if Angle and ReferenceAngle are apart my a lot (many multiples of 2*Pi),
//then you will get the wrong result. Thus, the intent is for both Angle and ReferenceAngle to be in the range [0, 2*Pi).
//Note that the output may or may not be in this range.
static double AdjustAngleToSameRange(double Angle, double ReferenceAngle) {
	double AnglePlus  = Angle + 2.0*PI;
	double AngleMinus = Angle - 2.0*PI;
	if (fabs(AnglePlus - ReferenceAngle) < fabs(Angle - ReferenceAngle))
		return AnglePlus;
	if (fabs(AngleMinus - ReferenceAngle) < fabs(Angle - ReferenceAngle))
		return AngleMinus;
	return Angle;
}

//Take a collection of (Lat, Lon) pairs and adjust Longitudes so that they do not straddle the longitude discontinuity.
//This may result in some latitudes being outside the principal range of [0, 2*PI).
static void AdjustLongitudesToCommonRange(Eigen::Vector2d & P1, Eigen::Vector2d & P2, Eigen::Vector2d & P3, Eigen::Vector2d & P4) {
	//We use the first point as our reference
	P2(1) = AdjustAngleToSameRange(P2(1), P1(1));
	P3(1) = AdjustAngleToSameRange(P3(1), P1(1));
	P4(1) = AdjustAngleToSameRange(P4(1), P1(1));
}

//Take a collection of (Lat, Lon) pairs and adjust Longitudes so that they do not straddle the longitude discontinuity.
//This may result in some latitudes being outside the principal range of [0, 2*PI).
static void AdjustLongitudesToCommonRange(Eigen::Vector2d & P1, Eigen::Vector2d & P2, Eigen::Vector2d & P3, Eigen::Vector2d & P4, Eigen::Vector2d & P5) {
	//We use the first point as our reference
	P2(1) = AdjustAngleToSameRange(P2(1), P1(1));
	P3(1) = AdjustAngleToSameRange(P3(1), P1(1));
	P4(1) = AdjustAngleToSameRange(P4(1), P1(1));
	P5(1) = AdjustAngleToSameRange(P5(1), P1(1));
}

// ****************************************************************************************************************************************
// ********************************************************   Component Classes   *********************************************************
// ****************************************************************************************************************************************
bool FRFVisualizationRGB::isRGB(void)           const { return true;  }
bool FRFVisualizationRGB::isColormap(void)      const { return false; }
bool FRFVisualizationColormap::isRGB(void)      const { return false; }
bool FRFVisualizationColormap::isColormap(void) const { return true;  }
void FRFVisualizationColormap::SortSetPoints(void)    { std::sort(SetPoints.begin(), SetPoints.end()); }

FRFLayer::FRFLayer(uint16_t Width, uint16_t Height) :
		UnitsCode(-1),
		alpha(1.0),
		beta(0.0),
		HasValidityMask(false),
		layerWidth(Width),
		layerHeight(Height),
		TypeCode(8U) {
	updateBitsPerSample();
}

uint8_t FRFLayer::GetTypeCode(void) const { return TypeCode; }

void FRFLayer::SetTypeCode(uint8_t NewTypeCode) {
	TypeCode = NewTypeCode;
	updateBitsPerSample();
}

void FRFLayer::updateBitsPerSample(void) {
	if ((TypeCode >= 1U) && (TypeCode <= 64U))
		bitsPerSample = (uint32_t) TypeCode;
	else if (TypeCode == 65U)
		bitsPerSample = 8U;
	else if (TypeCode == 66U)
		bitsPerSample = 16U;
	else if (TypeCode == 67U)
		bitsPerSample = 32U;
	else if (TypeCode == 68U)
		bitsPerSample = 64U;
	else if (TypeCode == 70U)
		bitsPerSample = 32U;
	else if (TypeCode == 71U)
		bitsPerSample = 64U;
	else {
		fprintf(stderr,"Error in FRFLayer::updateBitsPerSample - Invalid Type Code.\r\n");
		bitsPerSample = 0U;
	}
}

//This helper sets alpha and beta so that the lower end of the raw value range corresponds to MinVal and the upper end corresponds to MaxVal.
//If you change the data type for the layer you will to set alpha and beta again to get the desired range. If used on a floating-point layer,
//this sets alpha to 1 and beta to 0, effectively disabling the scale and offset since they are not generally necessary for float layers.
void FRFLayer::SetAlphaAndBetaForGivenRange(double MinVal, double MaxVal) {
	std::tuple<double, double> rawLimits = GetRawValueLimits();
	double minRawValue = std::get<0>(rawLimits);
	double maxRawValue = std::get<1>(rawLimits);
	
	if ((TypeCode >= 1U) && (TypeCode <= 68U)) {
		alpha = (MaxVal - MinVal)/(maxRawValue - minRawValue);
		beta = MinVal - alpha*minRawValue;
	}
	else if ((TypeCode == 70U) || (TypeCode == 71U)) {
		fprintf(stderr,"Warning in SetAlphaAndBetaForGivenRange: This method is not for float32 and float64 layers. Resetting alpha and beta.\r\n");
		alpha = 1.0;
		beta = 0.0;
	}
	else
		fprintf(stderr,"Error in FRFLayer::SetAlphaAndBetaForGivenRange - Invalid Type Code.\r\n");
}

//Get human-readable string indicating layer data type (e.g. 8-bit uint, 32-bit float)
std::string FRFLayer::GetTypeString(void) const {
	if ((TypeCode >= 1U) && (TypeCode <= 64U))
		return std::to_string(TypeCode) + std::string("-bit Unsigned Int");
	else if (TypeCode == 65U)
		return std::string("8-bit Signed Int");
	else if (TypeCode == 66U)
		return std::string("16-bit Signed Int");
	else if (TypeCode == 67U)
		return std::string("32-bit Signed Int");
	else if (TypeCode == 68U)
		return std::string("64-bit Signed Int");
	else if (TypeCode == 70U)
		return std::string("32-bit Float");
	else if (TypeCode == 71U)
		return std::string("64-bit Float");
	else
		return std::string("Invalid Type");
}

//Get lowest and highest values representable by layer, as configured. Returned as <lower, upper>
std::tuple<double, double> FRFLayer::GetRange(void) const {
	std::tuple<double, double> rawLimits = GetRawValueLimits();
	double minValue = alpha*std::get<0>(rawLimits) + beta;
	double maxValue = alpha*std::get<1>(rawLimits) + beta;
	return std::make_tuple(minValue, maxValue);
}

//Gets the difference between consecutive representable values. For Float data types this is not constant amongst all representable values. For values close to 0
//the difference is much smaller than it is for consecutive values far from 0. Thus, for float types we return NaN to mitigate the risk of misusing the result.
double FRFLayer::GetStepSize(void) const {
	if ((TypeCode >= 1U) && (TypeCode <= 68U))
		return fabs(alpha);
	else if ((TypeCode == 70U) || (TypeCode == 71U)) {
		fprintf(stderr,"Warning in GetStepSize: This method is not for float32 and float64 layers.\r\n");
		return std::nan("");
	}
	else {
		fprintf(stderr,"Error in FRFLayer::GetStepSize - Invalid Type Code.\r\n");
		return std::nan("");
	}
}

//Get a relatively tight upper bound on the quantization error introduced by our encoding. For relatively low-bit-count integer layers this will generally be a tight upper bound.
//For high-bit-depth integers, when the tolerance is smaller than the difference between consecutive normal 64-bit float numbers (DBL_EPSILON), we return 2*DBL_EPSILON. This ensures
//that if you compute the difference between the value set and the value retrieved at a pixel, the difference will be less than the tolerance returned by this method, although the
//tolerance is no longer a tight upper bound on quantization error. When comparing very-high-bit-depth integers, you should consider accessing the data layer directly instead of using
//the accessor methods provided by this class since converting to 64-bit floats may cause a loss of accuracy in such cases.
//Another warning: This is not a garenteed upper bound on error for float types. Due to the design of the floating point representation, the error depends on how big the number is.
//A worst-case error bound would be too large to be useful for anything. This method will return a worst-case upper bound on error for numbers in [-1, 1]. Outside this range, quantization
//error may exceed the value returned by this method.
double FRFLayer::GetTolerance(void) const {
	if (TypeCode == 70U)
		return alpha*((double) std::numeric_limits<float>::epsilon())/2.0 + 4.0*std::numeric_limits<double>::epsilon();
	else if (TypeCode == 71U)
		return alpha*std::numeric_limits<double>::epsilon() + 2.0*std::numeric_limits<double>::epsilon();
	else {
		double Tol = (GetStepSize()/2.0 + 5.0*std::numeric_limits<double>::epsilon());
		return std::max(Tol, 2.0*std::numeric_limits<double>::epsilon());
	}
}

//Helper function for range, tolerance, and coefficient-setting methods
std::tuple<double, double> FRFLayer::GetRawValueLimits(void) const {
	double minRawValue = 0.0;
	double maxRawValue = 0.0;
	if ((TypeCode >= 1U) && (TypeCode <= 64U))
		maxRawValue = pow(2.0, (double) TypeCode) - 1.0;
	else if (TypeCode == 65U) {
		minRawValue = (double) std::numeric_limits<int8_t>::lowest();
		maxRawValue = (double) std::numeric_limits<int8_t>::max();
	}
	else if (TypeCode == 66U) {
		minRawValue = (double) std::numeric_limits<int16_t>::lowest();
		maxRawValue = (double) std::numeric_limits<int16_t>::max();
	}
	else if (TypeCode == 67U) {
		minRawValue = (double) std::numeric_limits<int32_t>::lowest();
		maxRawValue = (double) std::numeric_limits<int32_t>::max();
	}
	else if (TypeCode == 68U) {
		minRawValue = (double) std::numeric_limits<int64_t>::lowest();
		maxRawValue = (double) std::numeric_limits<int64_t>::max();
	}
	else if (TypeCode == 70U) {
		minRawValue = (double) std::numeric_limits<float>::lowest();
		maxRawValue = (double) std::numeric_limits<float>::max();
	}
	else if (TypeCode == 71U) {
		minRawValue = std::numeric_limits<double>::lowest();
		maxRawValue = std::numeric_limits<double>::max();
	}
	else
		fprintf(stderr,"Error in FRFLayer::GetRawValueLimits - Invalid Type Code.\r\n");
	
	return std::make_tuple(minRawValue, maxRawValue);
}

//Get the number of bytes needed to store the raw value layer only (not including any validity mask)
uint64_t FRFLayer::GetTotalBytesWithoutValidityMask(void) const {
	uint64_t totalBits = ((uint64_t) layerWidth) * ((uint64_t) layerHeight) * ((uint64_t) bitsPerSample);
	return (totalBits + 7U) / 8U;
}

//Get the number of bytes needed to store the raw value layer and validity mask (if used)
uint64_t FRFLayer::GetTotalBytesWithValidityMask(void) const {
	uint64_t totalBitsForValueLayer = ((uint64_t) layerWidth) * ((uint64_t) layerHeight) * ((uint64_t) bitsPerSample);
	uint64_t totalBytesForValueLayer = (totalBitsForValueLayer + 7U) / 8U;
	uint64_t totalBitsForValidityMask = 0U;
	if (HasValidityMask)
		totalBitsForValidityMask = ((uint64_t) layerWidth) * ((uint64_t) layerHeight);
	uint64_t totalBytesForValidityMask = (totalBitsForValidityMask + 7U) / 8U;
	return totalBytesForValueLayer + totalBytesForValidityMask;
}

//Allocate memory for the image data and (if applicable) the validity mask
void FRFLayer::AllocateStorage(void) {
	Data.clear();
	Data.resize(GetTotalBytesWithValidityMask());
}

//Free memory for the image data and validity mask (setting values will fail and retrieval will return NaN if not allocated)
void FRFLayer::DeAllocateStorage(void) {
	Data.clear();
}

//Check pixel validity (returns true if there is no mask)
bool FRFLayer::GetValidityFromMask(uint32_t Row, uint32_t Col) const {
	//If there is no validity mask then all pixels are considered valid
	if (! HasValidityMask)
		return true;
	
	//If we have a validity mask, we need to check the correct bit in the mask
	uint32_t pixelNumber = Row*((uint32_t) layerWidth) + Col;
	uint64_t validityMaskByteOffset = GetTotalBytesWithoutValidityMask();
	uint64_t byteNumber = validityMaskByteOffset + (uint64_t) (pixelNumber / 8U);
	
	uint8_t maskByte = Data[byteNumber];
	switch ((uint8_t) (pixelNumber % 8U)) {
		case 0U: return ((maskByte & 0b10000000) != 0U);
		case 1U: return ((maskByte & 0b01000000) != 0U);
		case 2U: return ((maskByte & 0b00100000) != 0U);
		case 3U: return ((maskByte & 0b00010000) != 0U);
		case 4U: return ((maskByte & 0b00001000) != 0U);
		case 5U: return ((maskByte & 0b00000100) != 0U);
		case 6U: return ((maskByte & 0b00000010) != 0U);
		case 7U: return ((maskByte & 0b00000001) != 0U);
		default: return false; //This case should never happen
	};
}

//Mark a pixel as valid or invalid in mask. Does nothing if there is no mask.
void FRFLayer::SetValidityInMask(uint32_t Row, uint32_t Col, bool Valid) {
	//If there is no validity mask then issue warning since method does nothing
	if (! HasValidityMask) {
		fprintf(stderr,"Warning: Trying to modify validity mask that does not exist.\r\n");
		return;
	}
	
	//If we have a validity mask, we need to write to the correct bit in the mask
	uint32_t pixelNumber = Row*((uint32_t) layerWidth) + Col;
	uint64_t validityMaskByteOffset = GetTotalBytesWithoutValidityMask();
	uint64_t byteNumber = validityMaskByteOffset + (uint64_t) (pixelNumber / 8U);
	
	uint8_t bitNumber = (uint8_t) 7U - (uint8_t) (pixelNumber % 8U);
	uint8_t byteMask = ~(((uint8_t) 1U) << bitNumber);
	uint8_t contrib  = (Valid ? ((uint8_t) 1U) << bitNumber : (uint8_t) 0U);
	Data[byteNumber] = (Data[byteNumber] & byteMask) + contrib;
}

//Get the value of a pixel. This uses alpha and beta to correctly interpret stored data. An invalid pixel will return NaN.
double FRFLayer::GetValue(uint32_t Row, uint32_t Col) const {
	//Dispatch to the best getter for our data type
	return GetValue_Generic(Row, Col);
}

//Get the value of a pixel. This uses alpha and beta to correctly interpret stored data. An invalid pixel will return NaN.
double FRFLayer::GetValue_Generic(uint32_t Row, uint32_t Col) const {
	//If out of bounds, return NaN
	if ((Row >= layerHeight) || (Col >= layerWidth))
		return std::nan("");
	
	//If this layer uses an integer data type, check the validity mask first. If the pixel is invalid, we just return NaN.
	if ((TypeCode >= 1U) && (TypeCode <= 68U)) {
		if (! GetValidityFromMask(Row, Col))
			return std::nan("");
	}
	
	//Figure out which bits belong to this pixel
	uint64_t pixelNumber = Row*((uint64_t) layerWidth) + Col;
	uint64_t firstBit    = bitsPerSample*pixelNumber;
	uint64_t lastBit     = firstBit + bitsPerSample - 1U;
	
	//Grab these bits from our data buffer
	uint64_t bitPattern = grabBits(Data, firstBit, lastBit);
	
	//fprintf(stderr,"Pixel: %llu, First bit: %llu, Last bit: %llu\r\n", (unsigned long long) pixelNumber, (unsigned long long) firstBit, (unsigned long long) lastBit);
	//std::cerr << "Bit Pattern: " << std::bitset<64>(bitPattern) << "\r\n";
	
	//Now interpret the bit pattern and compute the value
	if ((TypeCode >= 1U) && (TypeCode <= 64U))
		return alpha*((double) bitPattern) + beta;
	if (TypeCode == 65U)
		return alpha*((double) ((int8_t) ((uint8_t) bitPattern))) + beta;
	if (TypeCode == 66U)
		return alpha*((double) ((int16_t) ((uint16_t) bitPattern))) + beta;
	if (TypeCode == 67U)
		return alpha*((double) ((int32_t) ((uint32_t) bitPattern))) + beta;
	if (TypeCode == 68U)
		return alpha*((double) ((int64_t) bitPattern)) + beta;
	if (TypeCode == 70U) {
		uint32_t shortPattern = (uint32_t) bitPattern;
		return alpha*((double) (reinterpret_cast<float &>(shortPattern))) + beta;
	}
	if (TypeCode == 71U)
		return alpha*(reinterpret_cast<double &>(bitPattern)) + beta;
	return 0.0;
}

//Set the value of a pixel. This uses alpha and beta to correctly encode the value. Set a pixel to NaN to mark it as invalid.
bool FRFLayer::SetValue(uint32_t Row, uint32_t Col, double Value) {
	//Dispatch to the best setter for our data type
	return SetValue_Generic(Row, Col, Value);
}

//Set the value of a pixel. This uses alpha and beta to correctly encode the value. Set a pixel to NaN to mark it as invalid.
bool FRFLayer::SetValue_Generic(uint32_t Row, uint32_t Col, double Value) {
	//If setting a pixel out of bounds, return false
	if ((Row >= layerHeight) || (Col >= layerWidth))
		return false;
	
	//For integer layers, we need to manage NaN and validity masks
	if ((TypeCode >= 1U) && (TypeCode <= 68U)) {
		if (std::isnan(Value)) {
			if (HasValidityMask) {
				SetValidityInMask(Row, Col, false);
				return true; //In this special case, we just need to mark the pixel as invalid - wo don't need to write anything to the data layer
			}
			else
				Value = 0.0; //We can't mark the pixel invalid - lets just write 0 to the data layer.
		}
		else if (HasValidityMask)
			SetValidityInMask(Row, Col, true);
	}
	
	//Undo the affine mapping
	Value = (Value - beta)/alpha;
	
	//fprintf(stderr,"Raw Value (unquantized): %f\r\n", Value);
	
	//Convert to a bit pattern for the appropriate value type
	uint64_t bitPattern = 0U;
	if ((TypeCode >= 1U) && (TypeCode <= 64U))
		bitPattern = saturateRound_uint(Value, TypeCode);
	else if (TypeCode == 65U)
		bitPattern = (uint64_t) ((uint8_t) saturateRound_int8(Value));
	else if (TypeCode == 66U)
		bitPattern = (uint64_t) ((uint16_t) saturateRound_int16(Value));
	else if (TypeCode == 67U)
		bitPattern = (uint64_t) ((uint32_t) saturateRound_int32(Value));
	else if (TypeCode == 68U)
		bitPattern = (uint64_t) saturateRound_int64(Value);
	else if (TypeCode == 70U) {
		//Warning: values outside the range of finite representable 32-bit floats will map to either the closest finite value or infinity.
		//Technically this is up to the implementation, but usually they will map to the closest finite value.
		float valueFloat32 = (float) Value;
		uint32_t shortPattern = reinterpret_cast<uint32_t &>(valueFloat32);
		bitPattern = (uint64_t) shortPattern;
	}
	else if (TypeCode == 71U)
		bitPattern = reinterpret_cast<uint64_t &>(Value);
	else
		return false;
	
	//std::cerr << "Bit Pattern: " << std::bitset<64>(bitPattern) << "\r\n";
	
	//Now, compute which bits we need to write this value to
	uint64_t pixelNumber = Row*((uint64_t) layerWidth) + Col;
	uint64_t firstBit    = bitsPerSample*pixelNumber;
	uint64_t lastBit     = firstBit + bitsPerSample - 1U;
	
	//fprintf(stderr,"pixelNumber: %llu\r\n", (unsigned long long) pixelNumber);
	//fprintf(stderr,"firstBit: %llu\r\n", (unsigned long long) firstBit);
	//fprintf(stderr,"lastBit: %llu\r\n", (unsigned long long) lastBit);
	
	//Write this bit pattern to our data buffer
	packBits(Data, firstBit, lastBit, bitPattern);
	return true;
}

//Grab a certain number of bits from a 4-byte bit pattern (stored in a uint32) and interpret as a signed 8-bit int. The result is returned as an int8.
//Msb is the highest bit and Lsb is the lowest bit to extract. [Lsb, Msb] should be 8-bits or less.
static int8_t getPackedInt8FromUint32(uint32_t BitPattern, uint8_t Lsb, uint8_t Msb) {
	uint32_t mask = (((uint32_t) 1U) << (Msb - Lsb + 1U)) - 1U;
	uint8_t myBits = (uint8_t) ((BitPattern >> Lsb) & mask);
	
	//Sign-extend to 8-bits, if the most significant bit is 1 and we are extracting fewer than 8 bits.
	if ((uint8_t) (myBits >> (Msb - Lsb)) > 0U) {
		uint8_t signExtension = 255U & ((uint8_t) ~mask);
		myBits += signExtension;
	}
	return (int8_t) myBits;
}

//Take a dimension string and an exponent and render a string of the form "", "str ", or "str^exp "... whichever is appropriate.
static std::string getComponentOfUnitsString(std::string BaseUnit, int8_t Exp) {
	if (Exp == 0)
		return std::string();
	else if (Exp == 1)
		return BaseUnit + " ";
	else
		return BaseUnit + "^" + std::to_string(Exp) + " ";
}

//Get a string showing units as a product of the SI base units with exponents, such as (kg m^2 s^-3)
std::string FRFLayer::GetUnitsStringInExponentForm(void) {
	bool   special = (((uint32_t) UnitsCode >> 31) == 1U);
	int8_t cd_exp  = getPackedInt8FromUint32((uint32_t) UnitsCode, 27U, 30U);
	int8_t mol_exp = getPackedInt8FromUint32((uint32_t) UnitsCode, 23U, 26U);
	int8_t K_exp   = getPackedInt8FromUint32((uint32_t) UnitsCode, 19U, 22U);
	int8_t A_exp   = getPackedInt8FromUint32((uint32_t) UnitsCode, 15U, 18U);
	int8_t s_exp   = getPackedInt8FromUint32((uint32_t) UnitsCode, 10U, 14U);
	int8_t kg_exp  = getPackedInt8FromUint32((uint32_t) UnitsCode,  5U,  9U);
	int8_t m_exp   = getPackedInt8FromUint32((uint32_t) UnitsCode,  0U,  4U);
	
	if (special) {
		if (UnitsCode == -1)
			return std::string("Unspecified");
		else
			return std::string("Unrecognized special units code");
	}
	
	std::string str;
	str += getComponentOfUnitsString(std::string("cd" ), cd_exp );
	str += getComponentOfUnitsString(std::string("mol"), mol_exp);
	str += getComponentOfUnitsString(std::string("K"  ), K_exp  );
	str += getComponentOfUnitsString(std::string("A"  ), A_exp  );
	str += getComponentOfUnitsString(std::string("kg" ), kg_exp );
	str += getComponentOfUnitsString(std::string("m"  ), m_exp  );
	str += getComponentOfUnitsString(std::string("s"  ), s_exp  );
	
	//Strip off trailing space, if there is one
	size_t endpos = str.find_last_not_of(" ");
	if (endpos == std::string::npos) str = std::string();
	else str = str.substr(0, endpos+1);
	
	if (str.empty())
		str = std::string("Dimensionless");
	return str;
}

//Get a string showing units in the most human-readable form
std::string FRFLayer::GetUnitsStringInMostReadableForm(void) {
	switch (UnitsCode) {
		case 31745: return std::string("m/s");
		case 30721: return std::string("m/s^2");
		case 62:    return std::string("kg/m^2");
		case 61:    return std::string("kg/m^3");
		case 29730: return std::string("Watts");
		case 30753: return std::string("Newtons");
		default:    return GetUnitsStringInExponentForm();
	}
}

//Retrieve the lat and lon of a grid point, by the row and column of the grid point.
Eigen::Vector2d FRFGeoRegistration::GetGridPointLatLon(uint16_t PointX, uint16_t PointY) const {
	uint32_t PointIndex = ((uint32_t) PointY) * (((uint32_t) GridWidthDivisor) + 1U) + ((uint32_t) PointX);
	if (PointIndex < GridPointLatLons.size())
		return GridPointLatLons[PointIndex];
	else
		return Eigen::Vector2d(std::nan(""), std::nan(""));
}

//Populate all fields except Altitude based on image corner locations. This results in a primitive Geo-Registration object with a single interpolation cell.
//The arguments UL, UR, LL, and LR are the Latitude-Longitude pairs (in radians) of the centers of the upper-left, upper-right, lower-left, and lower-right-most pixels in the image, respectively.
void FRFGeoRegistration::RegisterFromCornerLocations(Eigen::Vector2d const & UL, Eigen::Vector2d const & UR, Eigen::Vector2d const & LL, Eigen::Vector2d const & LR) {
	RegistrationType = 0U;
	GridWidthDivisor = 1U;
	GridHeightDivisor = 1U;
	GridPointLatLons.clear();
	GridPointLatLons.push_back(UL);
	GridPointLatLons.push_back(UR);
	GridPointLatLons.push_back(LL);
	GridPointLatLons.push_back(LR);
}


// ****************************************************************************************************************************************
// *********************************************************   FRF Image Class   **********************************************************
// ****************************************************************************************************************************************
//Create an empty FRF image
FRFImage::FRFImage() { Clear(); }

//Try to load an FRF image from disk. On failure, create an empty image
FRFImage::FRFImage(std::string Filepath) : FRFImage() { LoadFromDisk(Filepath); }

//Copying images is allowed, but each copy is deep and therefore expensive for large images, so use with caution
FRFImage::FRFImage(const FRFImage & Other) : FRFImage() {
	//Copy POD fields
	majorVersion    = Other.majorVersion;
	minorVersion    = Other.minorVersion;
	imageWidth      = Other.imageWidth;
	imageHeight     = Other.imageHeight;
	alphaLayerIndex = Other.alphaLayerIndex;
	
	//Copy dynamically managed fields
	Layers.resize(Other.Layers.size());
	for (size_t n = 0U; n < Other.Layers.size(); n++)
		Layers[n] = new FRFLayer(*(Other.Layers[n]));
	
	VisualizationManifest.resize(Other.VisualizationManifest.size());
	for (size_t n = 0U; n < Other.VisualizationManifest.size(); n++)
		VisualizationManifest[n] = Other.VisualizationManifest[n]->Clone();
	
	if (Other.GeoTagData != nullptr)
		GeoTagData = new FRFGeoTag(*(Other.GeoTagData));
	
	if (Other.GeoRegistrationData != nullptr)
		GeoRegistrationData = new FRFGeoRegistration(*(Other.GeoRegistrationData));
	
	//Copy Camera Info Tags and custom blocks containers
	CameraInformationTags = Other.CameraInformationTags;
	customBlocks = Other.customBlocks;
}

//Destructor
FRFImage::~FRFImage() { Clear(); }

//Load image from disk and populate this object. Returns true on success and false on failure.
//If loading fails for any reason, the object will be cleared before returning, so it is in the default-initialized state.
bool FRFImage::LoadFromDisk(std::string Filepath) {
	//Create a buffer to assemble file contents into
	std::vector<uint8_t> buffer;
	
	//Try to open the file for reading
	FILE * stream = fopen(Filepath.c_str(), "rb");
	if (stream == NULL) {
		fprintf(stderr,"File (%s) could not be opened for reading. Skipping load operation.\r\n", Filepath.c_str());
		Clear();
		return false;
	}
	
	//Read the first 16 bytes of the file header and decode them
	buffer.resize(16U);
	if (fread((void *) &(buffer[0U]), 1U, 16U, stream) != 16U) {
		fprintf(stderr,"Error in FRFImage::LoadFromDisk. Not enough bytes in file. Aborting load.\r\n");
		fclose(stream);
		Clear();
		return false;
	}
	auto iter = buffer.cbegin();
	uint64_t MagicNumber = decodeField_uint64(iter);
	majorVersion = decodeField_uint16(iter);
	minorVersion = decodeField_uint16(iter);
	imageWidth   = decodeField_uint16(iter);
	imageHeight  = decodeField_uint16(iter);
	if (MagicNumber != (uint64_t) 3197395143525533696U) {
		fprintf(stderr,"Error in FRFImage::LoadFromDisk. Magic number is not correct - Corrupt file? Aborting load.\r\n");
		fclose(stream);
		Clear();
		return false;
	}
	
	//Now read information blocks until we have decoded the entire header (we will know because of the End-Of-Header block)
	while (true) {
		buffer.resize(6U);
		if (fread((void *) &(buffer[0U]), 1U, 6U, stream) != 6U) {
			fprintf(stderr,"Error in FRFImage::LoadFromDisk. Not enough bytes in file. Aborting load.\r\n");
			fclose(stream);
			Clear();
			return false;
		}
		iter = buffer.cbegin();
		uint16_t BlockCode = decodeField_uint16(iter);
		uint32_t Size      = decodeField_uint32(iter);
		
		if (Size < 6U) {
			fprintf(stderr,"Error in FRFImage::LoadFromDisk. Size field in header info block has value less than 6. Aborting load.\r\n");
			fclose(stream);
			Clear();
			return false;
		}
		buffer.resize(Size - 6U);
		if (Size > 6U) {
			//We have a non-empty payload - read it into a buffer
			if (fread((void *) &(buffer[0U]), 1U, (size_t) (Size - 6U), stream) != (size_t) (Size - 6U)) {
				fprintf(stderr,"Error in FRFImage::LoadFromDisk. Not enough bytes in file. Aborting load.\r\n");
				fclose(stream);
				Clear();
				return false;
			}
		}
		
		//Now, the buffer holds the block payload, even if it is empty
		if (BlockCode == 0U) {
			//Decode Layer Manifest Block
			if (buffer.size() < 32U) {
				fprintf(stderr,"Error in FRFImage::LoadFromDisk. Too few bytes in layer manifest payload. Aborting load.\r\n");
				fclose(stream);
				Clear();
				return false;
			}
			iter = buffer.cbegin();
			alphaLayerIndex = decodeField_uint16(iter);
			unsigned int bytesLeft = ((unsigned int) buffer.size()) - 2U;
			
			while (bytesLeft > 0U) {
				FRFLayer * newLayer = this->AddLayer();
				deserialize_LayerManifestSection(newLayer, iter, bytesLeft);
			}
			
		}
		else if (BlockCode == 1U) {
			//Decode Visualizations Block
			if (buffer.size() < 16U) {
				fprintf(stderr,"Error in FRFImage::LoadFromDisk. Too few bytes in visualizations block payload. Aborting load.\r\n");
				fclose(stream);
				Clear();
				return false;
			}
			iter = buffer.cbegin();
			unsigned int bytesLeft = (unsigned int) buffer.size();
			while (bytesLeft > 0U) {
				FRFVisualization * vis = deserialize_VisualizationSection(iter, bytesLeft);
				VisualizationManifest.push_back(vis);
			}
		}
		else if (BlockCode == 2U) {
			//Decode Geo-Tagging Block
			if (buffer.size() != 108U)
				fprintf(stderr,"Error in FRFImage::LoadFromDisk. Incorrect payload size for Geo-Tagging block. Dropping block.\r\n");
			else {
				iter = buffer.cbegin();
				GeoTagData = deserialize_GeoTagPayload(iter);
			}
		}
		else if (BlockCode == 3U) {
			//Decode Geo-Registration Block
			iter = buffer.cbegin();
			unsigned int payloadSize = (unsigned int) buffer.size();
			GeoRegistrationData = deserialize_GeoRegistrationPayload(iter, payloadSize);
		}
		else if (BlockCode == 4U) {
			//Decode Camera Information Block
			iter = buffer.cbegin();
			unsigned int bytesLeft = (unsigned int) buffer.size();
			while (bytesLeft > 0U) {
				if (bytesLeft < 4U) {
					fprintf(stderr,"Warning in FRFImage::LoadFromDisk. Dropping partial Cam Info Tag.\r\n");
					break;
				}
				
				uint16_t TagCode = decodeField_uint16(iter);
				uint16_t TagPayloadSize = decodeField_uint16(iter);
				bytesLeft -= 4U;
				if (TagPayloadSize > bytesLeft) {
					fprintf(stderr,"Warning in FRFImage::LoadFromDisk. Dropping partial Cam Info Tag.\r\n");
					break;
				}
				
				//Add this camera info tag
				std::vector<uint8_t> & TagData = CameraInformationTags[TagCode];
				TagData.clear();
				TagData.insert(TagData.begin(), iter, iter + TagPayloadSize);
				bytesLeft -= TagPayloadSize;
			}
		}
		else if (BlockCode == 5U) {
			//Decode Custom Block
			if (buffer.size() < 8U)
				fprintf(stderr,"Error in FRFImage::LoadFromDisk. Custom block payload too small. Dropping block.\r\n");
			else {
				iter = buffer.cbegin();
				uint64_t CustomBlockCode = decodeField_uint64(iter);
				FRFCustomBlock & block = customBlocks[CustomBlockCode];
				block.CustomBlockCode = CustomBlockCode;
				block.BlockPayload.clear();
				block.BlockPayload.insert(block.BlockPayload.begin(), iter, buffer.cend());
			}
		}
		else if (BlockCode == 6U) {
			//Process End-Of-Header block
			break;
		}
		else
			fprintf(stderr,"Warning: Dropping block with unrecognized BlockCode.\r\n");
	}
	
	//Now that the header has been read, read in the image data and populate our layer objects
	for (FRFLayer * layer : Layers) {
		layer->AllocateStorage();
		uint64_t totalLayerSize = layer->GetTotalBytesWithValidityMask();
		if (fread((void *) &(layer->Data[0U]), 1U, totalLayerSize, stream) != totalLayerSize) {
			fprintf(stderr,"Error in FRFImage::LoadFromDisk. Not enough bytes in file. Aborting load.\r\n");
			fclose(stream);
			Clear();
			return false;
		}
	}
	
	//We should now have read in all bytes from the file. We check to see if there are any bytes left and issue a warning if we find any.
	uint8_t temp = 0U;
	if (fread((void *) &temp, 1U, 1U, stream) == 1U)
		fprintf(stderr,"Warning in FRFImage::LoadFromDisk. There are extra bytes left in the file (after load). File may be corrupt.\r\n");
	
	//Close the file and return success
	fclose(stream);
	return true;
}

//Save image to disk. Returns true on success and false on failure.
bool FRFImage::SaveToDisk(std::string Filepath) const {
	//First check to make sure the image object is complete and valid. We will not create an invalid file.
	if (Layers.empty()) {
		fprintf(stderr,"Error in FRFImage::SaveToDisk. Can not save image with 0 layers.\r\n");
		return false;
	}
	if (VisualizationManifest.empty()) {
		fprintf(stderr,"Error in FRFImage::SaveToDisk. Can not save image with 0 visualizations.\r\n");
		return false;
	}
	for (const FRFLayer * layer : Layers) {
		if (layer->Data.size() != layer->GetTotalBytesWithValidityMask()) {
			fprintf(stderr,"Error in FRFImage::SaveToDisk. Image has layer with incorrect buffer size. Aborting.\r\n");
			return false;
		}
	}
	
	//Create a buffer to assemble file contents into
	std::vector<uint8_t> buffer;
	
	FILE * stream = fopen(Filepath.c_str(), "wb");
	if (stream == NULL) {
		fprintf(stderr,"Error in FRFImage::SaveToDisk. Could not open file for writing: %s\r\n", Filepath.c_str());
		return false;
	}
	
	//Encode File Header
	encodeField_uint64 (buffer, (uint64_t) 3197395143525533696U);
	encodeField_uint16 (buffer, majorVersion);
	encodeField_uint16 (buffer, minorVersion);
	encodeField_uint16 (buffer, imageWidth);
	encodeField_uint16 (buffer, imageHeight);
	
	//Encode Layer Manifest Block
	std::vector<uint8_t> blockPayload;
	encodeField_uint16 (blockPayload, alphaLayerIndex);
	for (const FRFLayer * layer : Layers)
		serialize_LayerManifestSection(layer, blockPayload);
	encodeField_uint16 (buffer, 0U);                                             //Encode block code
	encodeField_uint32 (buffer, (uint32_t) blockPayload.size() + (uint32_t) 6U); //Encode block size
	buffer.insert(buffer.end(), blockPayload.begin(), blockPayload.end());       //Insert block payload
	
	//Encode Visualizations Block
	blockPayload.clear();
	for (const FRFVisualization * viz : VisualizationManifest)
		serialize_VisualizationSection(viz, blockPayload);
	encodeField_uint16 (buffer, 1U);                                             //Encode block code
	encodeField_uint32 (buffer, (uint32_t) blockPayload.size() + (uint32_t) 6U); //Encode block size
	buffer.insert(buffer.end(), blockPayload.begin(), blockPayload.end());       //Insert block payload
	
	//Encode Geo-Tagging Block (if used)
	if (GeoTagData != NULL) {
		encodeField_uint16 (buffer, 2U);                  //Encode block code
		encodeField_uint32 (buffer, (uint32_t) 114U);     //Encode block size
		serialize_GeoTagBlockPayload(GeoTagData, buffer); //Encode block payload
	}
	
	//Encode Geo-Registration Block (if used)
	if (GeoRegistrationData != NULL) {
		blockPayload.clear();
		serialize_GeoRegistrationPayload(GeoRegistrationData, blockPayload);
		encodeField_uint16 (buffer, 3U);                                             //Encode block code
		encodeField_uint32 (buffer, (uint32_t) blockPayload.size() + (uint32_t) 6U); //Encode block size
		buffer.insert(buffer.end(), blockPayload.begin(), blockPayload.end());       //Insert block payload
	}
	
	//Encode Camera Information Block (if used)
	if (! CameraInformationTags.empty()) {
		blockPayload.clear();
		for (const auto & kv : CameraInformationTags) {
			encodeField_uint16 (blockPayload, kv.first);
			encodeField_uint16 (blockPayload, (uint16_t) kv.second.size());
			blockPayload.insert(blockPayload.end(), kv.second.begin(), kv.second.end());
		}
		encodeField_uint16 (buffer, 4U);                                             //Encode block code
		encodeField_uint32 (buffer, (uint32_t) blockPayload.size() + (uint32_t) 6U); //Encode block size
		buffer.insert(buffer.end(), blockPayload.begin(), blockPayload.end());       //Insert block payload
	}
	
	//Encode Custom Blocks
	for (const auto & kv : customBlocks) {
		blockPayload.clear();
		serialize_CustomBlockPayload(kv.second, blockPayload);
		encodeField_uint16 (buffer, 5U);                                             //Encode block code
		encodeField_uint32 (buffer, (uint32_t) blockPayload.size() + (uint32_t) 6U); //Encode block size
		buffer.insert(buffer.end(), blockPayload.begin(), blockPayload.end());       //Insert block payload
	}
	
	//Encode End-of-Header block
	encodeField_uint16 (buffer, 6U);            //Encode block code
	encodeField_uint32 (buffer, (uint32_t) 6U); //Encode block size
	
	//Save FRF header to file
	if (fwrite((void *) &(buffer[0]), 1, buffer.size(), stream) != buffer.size()) {
		fprintf(stderr,"Error in FRFImage::SaveToDisk: Could not write all bytes to file stream.\r\n");
		fclose(stream);
		return false;
	}
	buffer.clear(); //Free up buffer for FRF header
	
	//Save layers to file
	for (const FRFLayer * layer : Layers) {
		if (fwrite((void *) &(layer->Data[0]), 1, layer->Data.size(), stream) != layer->Data.size()) {
			fprintf(stderr,"Error in FRFImage::SaveToDisk: Could not write all bytes to file stream.\r\n");
			fclose(stream);
			return false;
		}
	}
	
	//Close file and return success
	fclose(stream);
	return true;
}

//Return <Major, Minor> version tuple for the image
std::tuple<uint16_t, uint16_t> FRFImage::getFileVersion(void) const {
	return std::make_tuple(majorVersion, minorVersion);
}

//Destroy image and reset object to default-initialized state
void FRFImage::Clear(void) {
	majorVersion    = 0U;
	minorVersion    = 1U;
	imageWidth      = 0U;
	imageHeight     = 0U;
	alphaLayerIndex = std::numeric_limits<uint16_t>::max();
	
	for (FRFLayer * layer : Layers)
		delete layer;
	Layers.clear();
	
	for (FRFVisualization * vis : VisualizationManifest)
		delete vis;
	VisualizationManifest.clear();
	
	if (GeoTagData != NULL)
		delete GeoTagData;
	GeoTagData = NULL;
	
	if (GeoRegistrationData != NULL)
		delete GeoRegistrationData;
	GeoRegistrationData = NULL;
	
	CameraInformationTags.clear();
	customBlocks.clear();
}

//Get image width (pixels)
uint16_t FRFImage::Width(void) const {
	return imageWidth;
}

//Get image height (pixels)
uint16_t FRFImage::Height(void) const {
	return imageHeight;
}

//Same as Width()
uint16_t FRFImage::Cols(void) const {
	return imageWidth;
}

//Same as Height()
uint16_t FRFImage::Rows(void) const {
	return imageHeight;
}

//Set image width (pixels). Can only be performed on an empty image (no layers).
bool FRFImage::SetWidth(uint16_t Width) {
	if (Layers.empty()) {
		imageWidth = Width;
		return true;
	}
	else
		return false;
}

//Set image height (pixels). Can only be performed on an empty image (no layers).
bool FRFImage::SetHeight(uint16_t Height) {
	if (Layers.empty()) {
		imageHeight = Height;
		return true;
	}
	else
		return false;
}

//Return number of layers in image
uint16_t FRFImage::NumberOfLayers(void) const {
	return ((uint16_t) Layers.size());
}

//Get const access to a layer for reading
FRFLayer const * FRFImage::Layer(uint16_t LayerIndex) const {
	if ((size_t) LayerIndex < Layers.size())
		return Layers[LayerIndex];
	else
		return NULL;
}

//Get access to a layer with permission to modify
FRFLayer * FRFImage::Layer(uint16_t LayerIndex) {
	if ((size_t) LayerIndex < Layers.size())
		return Layers[LayerIndex];
	else
		return NULL;
}

//Add a new layer to the image and return a pointer to the new layer (with permission to modify)
FRFLayer * FRFImage::AddLayer(void) {
	Layers.push_back(new FRFLayer(imageWidth, imageHeight));
	return Layers.back();
}

//Delete a layer from the image.
void FRFImage::RemoveLayer(uint16_t LayerIndex) {
	if ((size_t) LayerIndex < Layers.size()) {
		delete Layers[LayerIndex];
		Layers.erase(Layers.begin() + LayerIndex);
	}
}

//Returns true if the image has an alpha layer and false otherwise
bool FRFImage::HasAlphaLayer(void) const { return (alphaLayerIndex != std::numeric_limits<uint16_t>::max()); }

//Get the index of the alpha layer (returns 2^16-1 if there is no alpha layer)
uint16_t FRFImage::GetAlphaLayerIndex(void) const { return alphaLayerIndex; }

//Set the alpha layer index
void FRFImage::SetAlphaLayerIndex(uint16_t Index) { alphaLayerIndex = Index; }

//Clear the alpha layer designation (this does not destroy any layers - it just unmarks the alpha layer, if there is one)
void FRFImage::RemoveAlphaLayer(void) { alphaLayerIndex = std::numeric_limits<uint16_t>::max(); }

//Get number of visualizations
uint16_t FRFImage::GetNumberOfVisualizations(void) const {
	return ((uint16_t) VisualizationManifest.size());
}

//Get const access to a layer for reading
FRFVisualization const * FRFImage::Visualization(uint16_t VisualizationIndex) const {
	if ((size_t) VisualizationIndex < VisualizationManifest.size())
		return VisualizationManifest[VisualizationIndex];
	else
		return NULL;
}

//Get access to a layer with permission to modify
FRFVisualization * FRFImage::Visualization(uint16_t VisualizationIndex) {
	if ((size_t) VisualizationIndex < VisualizationManifest.size())
		return VisualizationManifest[VisualizationIndex];
	else
		return NULL;
}

//Add a new RGB visualization and return pointer to it (with permission to modify)
FRFVisualizationRGB * FRFImage::AddVisualizationRGB(void) {
	FRFVisualizationRGB * vis = new FRFVisualizationRGB;
	VisualizationManifest.push_back(dynamic_cast<FRFVisualization *>(vis));
	return vis;
}

//Add a new Colormap visualization and return pointer to it (with permission to modify)
FRFVisualizationColormap * FRFImage::AddVisualizationColormap(void) {
	FRFVisualizationColormap * vis = new FRFVisualizationColormap;
	VisualizationManifest.push_back(dynamic_cast<FRFVisualization *>(vis));
	return vis;
}

//Delete a visualization
void FRFImage::RemoveVisualization(uint16_t VisualizationIndex) {
	if ((size_t) VisualizationIndex < VisualizationManifest.size()) {
		delete VisualizationManifest[VisualizationIndex];
		VisualizationManifest.erase(VisualizationManifest.begin() + VisualizationIndex);
	}
}

//Returns true if the object has a GeoTag and false otherwise
bool FRFImage::HasGeoTag(void) const {
	if (GeoTagData != NULL)
		return true;
	else
		return false;
}

//Get const pointer to GeoTag. Returns NULL if there is no GeoTag
FRFGeoTag const * FRFImage::GetGeoTag(void) const {
	return GeoTagData;
}

//Get regular pointer to GeoTag. Returns NULL if there is no GeoTag
FRFGeoTag * FRFImage::GetGeoTag(void) {
	return GeoTagData;
}

//Set the GeoTag
void FRFImage::SetGeoTag(FRFGeoTag GeoTag) {
	GeoTagData = new FRFGeoTag(GeoTag);
}

//Get rid of GeoTag, if there is one
void FRFImage::RemoveGeoTag(void) {
	if (GeoTagData != NULL) {
		delete GeoTagData;
		GeoTagData = NULL;
	}
}

//Returns true if the object has GeoRegistration data and false otherwise
bool FRFImage::IsGeoRegistered(void) const {
	if (GeoRegistrationData != NULL)
		return true;
	else
		return false;
}

//Get const access to GeoRegistration data. Returns NULL if non-existent
FRFGeoRegistration const * FRFImage::GetGeoRegistration(void) const {
	return GeoRegistrationData;
}

//Get access to GeoRegistration data. Returns NULL if non-existent
FRFGeoRegistration * FRFImage::GetGeoRegistration(void) {
	return GeoRegistrationData;
}

//Set the GeoRegistration data
void FRFImage::SetGeoRegistration(FRFGeoRegistration GeoRegistration) {
	GeoRegistrationData = new FRFGeoRegistration(GeoRegistration);
}

//Get rid of the GeoRegistration data, if there is any
void FRFImage::RemoveGeoRegistration(void) {
	if (GeoRegistrationData != NULL) {
		delete GeoRegistrationData;
		GeoRegistrationData = NULL;
	}
}

//Returns <Lat, Lon> tuple (both in radians) for the center of a specific pixel
std::tuple<double, double> FRFImage::GetCoordinatesOfPixel(uint16_t Row, uint16_t Col) const {
	return GetCoordinatesOfPixel(Eigen::Vector2d(Col, Row));
}

//Returns <Lat, Lon> tuple (both in radians) for arbitrary (x,y) pixel coordinates
std::tuple<double, double> FRFImage::GetCoordinatesOfPixel(Eigen::Vector2d const & PixelCoords) const {
	if (! IsGeoRegistered())
		return std::make_tuple(std::nan(""), std::nan(""));
	if (GeoRegistrationData->RegistrationType != 0U) {
		fprintf(stderr,"Error: Unrecognized registration type.\r\n");
		return std::make_tuple(std::nan(""), std::nan(""));
	}
	
	double pixelX = PixelCoords(0);
	double pixelY = PixelCoords(1);
	
	//Figure out which grid cell the pixel is in
	double cellWidth  = (((double) imageWidth ) - 1.0) / ((double) GeoRegistrationData->GridWidthDivisor );
	double cellHeight = (((double) imageHeight) - 1.0) / ((double) GeoRegistrationData->GridHeightDivisor);
	int cellX = (int) floor(pixelX/cellWidth );
	int cellY = (int) floor(pixelY/cellHeight);
	cellX = std::max(std::min(cellX, ((int) GeoRegistrationData->GridWidthDivisor ) - 1), 0);
	cellY = std::max(std::min(cellY, ((int) GeoRegistrationData->GridHeightDivisor) - 1), 0);
	
	double xMin = ((double)  cellX     ) * cellWidth;
	double xMax = ((double) (cellX + 1)) * cellWidth;
	double yMin = ((double)  cellY     ) * cellHeight;
	double yMax = ((double) (cellY + 1)) * cellHeight;
	
	//Retrieve the geo-locations of the 4 grid points surrounding our cell
	Eigen::Vector2d P1Value = GeoRegistrationData->GetGridPointLatLon((uint16_t)  cellX,      (uint16_t)  cellY     ); //Upper-left  grid point
	Eigen::Vector2d P2Value = GeoRegistrationData->GetGridPointLatLon((uint16_t) (cellX + 1), (uint16_t)  cellY     ); //Upper-right grid point
	Eigen::Vector2d P3Value = GeoRegistrationData->GetGridPointLatLon((uint16_t)  cellX,      (uint16_t) (cellY + 1)); //Lower-left  grid point
	Eigen::Vector2d P4Value = GeoRegistrationData->GetGridPointLatLon((uint16_t) (cellX + 1), (uint16_t) (cellY + 1)); //Lower-right grid point
	//fprintf(stderr, "P1: %f, %f\r\n", P1Value(0), P1Value(1));
	//fprintf(stderr, "P2: %f, %f\r\n", P2Value(0), P2Value(1));
	//fprintf(stderr, "P3: %f, %f\r\n", P3Value(0), P3Value(1));
	//fprintf(stderr, "P4: %f, %f\r\n", P4Value(0), P4Value(1));
	AdjustLongitudesToCommonRange(P1Value, P2Value, P3Value, P4Value); //Map all longitudes to the same side of the discontinuity
	//fprintf(stderr, "P1: %f, %f\r\n", P1Value(0), P1Value(1));
	//fprintf(stderr, "P2: %f, %f\r\n", P2Value(0), P2Value(1));
	//fprintf(stderr, "P3: %f, %f\r\n", P3Value(0), P3Value(1));
	//fprintf(stderr, "P4: %f, %f\r\n", P4Value(0), P4Value(1));
	//fprintf(stderr, "\r\n");
	
	//Use bilinear interpolation on these 4 grid points to compute the geo-location of our point
	double t = (xMax - pixelX)/cellWidth;
	Eigen::Vector2d P5Value = t*P1Value + (1.0 - t)*P2Value;
	Eigen::Vector2d P6Value = t*P3Value + (1.0 - t)*P4Value;
	double s = (yMax - pixelY)/cellHeight;
	Eigen::Vector2d LatLon = s*P5Value + (1.0 - s)*P6Value;
	LatLon = MapLatLonToPrincipalRange(LatLon);
	
	return std::make_tuple(LatLon(0), LatLon(1));
}

//Return the Z component of the cross product of V1 and V2 (V1 x V2), each viewed as a vector in R^3 with Z-component 0.
static double cross2(Eigen::Vector2d V1, Eigen::Vector2d V2) {
	return (V1(0)*V2(1) - V1(1)*V2(0));
}

//Let f(v) be the bi-linear interpolation function that maps (x1, y1) -> C1, (x2, y1) -> C2, (x2, y2) -> C3, and (x1, y2) -> C4
//Take the normalized x-value of a point mapping to P = (u, v) through this function and compute the normalized y-value of this point.
//This is a helper function for InverseBilinearInterp().
static double ybarFromXbar(double xbar, double u, Eigen::Vector2d C1, Eigen::Vector2d C2, Eigen::Vector2d C3, Eigen::Vector2d C4) {
	double num = u - C1(0) - (C2(0) - C1(0))*xbar;
	double den = (C4(0) - C1(0)) + (C1(0) - C2(0) + C3(0) - C4(0))*xbar;
	if (fabs(den) < 1e-13)
		return std::nan("");
	else
		return num/den;
}


//Return the distance between the point V and a given rectange (this is inf(|V - P|) over all P in [x1,x2] x [y1,y2])
//Notice that since [x1,x2] x [y1,y2] is a compact set, this is the same as min(|V - P|) over all P in [x1,x2] x [y1,y2].
static double distFromRectange(Eigen::Vector2d V, double x1, double x2, double y1, double y2) {
	double x = V(0);
	double y = V(1);
	
	if (x < x1) {
		if (y < y1)
			return (V - Eigen::Vector2d(x1, y1)).norm();
		else if (y <= y2)
			return (x1 - x);
		else
			return (V - Eigen::Vector2d(x1, y2)).norm();
	}
	else if (x <= x2) {
		if (y < y1)
			return (y1 - y);
		else if (y <= y2)
			return 0.0;
		else
			return (y - y2);
	}
	else {
		if (y < y1)
			return (V - Eigen::Vector2d(x2, y1)).norm();
		else if (y <= y2)
			return (x - x2);
		else
			return (V - Eigen::Vector2d(x2, y2)).norm();
	}
}

//Let f(x) be the bilinear interpolation function  that maps (x1, y1) -> C1, (x2, y1) -> C2, (x2, y2) -> C3, and (x1, y2) -> C4.
//Invert this function and find the point v s.t. f(v) = P. Notice that depending on the configuration of the 4 points C1-C4, there may be 0, 1, or 2 points that
//map to P. The "typical" case (where the points are in general position) has 2 solutions for almost all points P in the range of the interpolation function. In this case we select the
//candidate that is closest to the center of the square [x1, x2] x [y1, y2]. Another common case is that the points C1-C4 form a parallelogram. In this case the interpolation function
//is bijective and there is exactly one point in the pre-image of any given P. There are some cases (especially with degenerate configurations of C1-C4) where the pre-image for a point P
//is empty. In this case, we return a vector of NaNs.
static Eigen::Vector2d InverseBilinearInterp(Eigen::Vector2d P, double x1, double x2, double y1, double y2, Eigen::Vector2d C1, Eigen::Vector2d C2, Eigen::Vector2d C3, Eigen::Vector2d C4) {
	double u = P(0);
	double v = P(1);
	double a = cross2(C3, C2) + cross2(C4, C1) + cross2(C1, C3) + cross2(C2, C4);
	double b = cross2(C3, C1) + 2*cross2(C1, C4) + cross2(C4, C2) + (C1(1) - C2(1) + C3(1) - C4(1))*u - (C1(0) - C2(0) + C3(0) - C4(0))*v;
	double c = cross2(C4, C1) + (C4(1) - C1(1))*u - (C4(0) - C1(0))*v;
	
	if (fabs(a) + fabs(b) + fabs(c) < 1e-20)
		fprintf(stderr,"Warning: Degenerate interpolation function.\r\n");
	else {
		double s = 1.0/(fabs(a) + fabs(b) + fabs(c));
		a = a*s;
		b = b*s;
		c = c*s;
	}
	
	double xbar = 0.0;
	double ybar = 0.0;
	if (fabs(a) < 1e-13) {
		if (fabs(b) < 1e-13) {
			//There is no solution
			xbar = std::nan("");
			ybar = std::nan("");
		}
		else {
			//The interpolation function is injective everywhere - we have exactly 1 solution
			xbar = -1.0*c/b;
			ybar = ybarFromXbar(xbar, u, C1, C2, C3, C4);
		}
	}
	else {
		//The interpolation function is non-injective. We will have 0-2 solutions
		double desc = b*b - 4.0*a*c;
		if (desc < 0.0) {
			xbar = std::nan("");
			ybar = std::nan("");
		}
		else if (desc < 1e-13) {
			xbar = (-1.0*b)/(2.0*a);
			ybar = ybarFromXbar(xbar, u, C1, C2, C3, C4);
		}
		else {
			//Compute the two candidate xbars. They are distinct in this case
			double xbar1 = (-1.0*b + sqrt(desc))/(2.0*a);
			double xbar2 = (-1.0*b - sqrt(desc))/(2.0*a);
		
			double ybar1 = ybarFromXbar(xbar1, u, C1, C2, C3, C4);
			double ybar2 = ybarFromXbar(xbar2, u, C1, C2, C3, C4);
			
			double d1 = distFromRectange(Eigen::Vector2d(xbar1, ybar1), 0.0, 1.0, 0.0, 1.0);
			double d2 = distFromRectange(Eigen::Vector2d(xbar2, ybar2), 0.0, 1.0, 0.0, 1.0);
			
			if ((d1 > 1e-13) && (d2 > 1e-13)) {
				//Both points in pre-image are outside of unit square - choose the one that is closest in case this is just due to round-off error
				if (d1 < d2) {
					xbar = xbar1;
					ybar = ybar1;
				}
				else {
					xbar = xbar2;
					ybar = ybar2;
				}
			}
			else if ((d1 <= 1e-13) && (d2 > 1e-13)) {
				//Candidate 1 is the only one in the unit square
				xbar = xbar1;
				ybar = ybar1;
			}
			else if ((d1 > 1e-13) && (d2 <= 1e-13)) {
				//Candidate 2 is the only one in the unit square
				xbar = xbar2;
				ybar = ybar2;
			}
			else {
				//Both candidates lie in the interior of the unit square. We return the one closest to the center, in case this is due to round-off error.
				fprintf(stderr,"Warning: Interpolation function is non-injective here. Two points in pre-image!");
				Eigen::Vector2d W1 = Eigen::Vector2d(xbar1, ybar1) - Eigen::Vector2d(0.5, 0.5);
				Eigen::Vector2d W2 = Eigen::Vector2d(xbar2, ybar2) - Eigen::Vector2d(0.5, 0.5);
				if (W1.norm() <= W2.norm()) {
					xbar = xbar1;
					ybar = ybar1;
				}
				else {
					xbar = xbar2;
					ybar = ybar2;
				}
				
			}
		}
	}

	double x = xbar*(x2 - x1) + x1;
	double y = ybar*(y2 - y1) + y1;
	return Eigen::Vector2d(x, y);
}

//Returns (x,y) pixel coordinates corresponding to a given Lat, Lon location.
Eigen::Vector2d FRFImage::GetPixelCoordsForLocation(double Lat, double Lon) const {
	//Right now, we do a naive search through each cell in the interpolation grid. If we find a cell where the local inverse lies within the cell, we stop the search
	//and go with that value. Otherwise, we finish searching all cells and use the value that was closest to the cell that generated it. For images with lots of interpolation
	//cells this is not very fast. A smarter strategy would be to use a good initial guess as to which cell the solution lies in (pehaps by viewing the whole image as one cell
	//using the outer corners of the image and the grid points and inverting), and then doing a spiral search starting from that cell. However, the usual case will be relatively
	//few cells so we will deal with that later if necessary.
	if (! IsGeoRegistered())
		return Eigen::Vector2d(std::nan(""), std::nan(""));
	if (GeoRegistrationData->RegistrationType != 0U) {
		fprintf(stderr,"Error: Unrecognized registration type.\r\n");
		return Eigen::Vector2d(std::nan(""), std::nan(""));
	}
	
	Eigen::Vector2d P = MapLatLonToPrincipalRange(Eigen::Vector2d(Lat, Lon)); //Vectorized form of point we are finding inverse for (with Longitude in principal range)
	
	double cellWidth  = (((double) imageWidth ) - 1.0) / ((double) GeoRegistrationData->GridWidthDivisor );
	double cellHeight = (((double) imageHeight) - 1.0) / ((double) GeoRegistrationData->GridHeightDivisor);
	uint16_t GridWidth  = GeoRegistrationData->GridWidthDivisor;  //Number of columns in the interpolation grid (grid of cells, not tie points)
	uint16_t GridHeight = GeoRegistrationData->GridHeightDivisor; //Number of rows in the interpolation grid (grid of cells, not tie points)
	
	Eigen::Vector2d v(std::nan(""), std::nan("")); //This will hold the final inverse of the interpolation function
	double distFromParentCell = std::numeric_limits<double>::infinity();
	
	for (uint16_t cellY = 0U; cellY < GridHeight; cellY++) {
		for (uint16_t cellX = 0U; cellX < GridWidth; cellX++) {
			//Retrieve the geo-locations of the 4 grid points surrounding our cell
			Eigen::Vector2d C1 = GeoRegistrationData->GetGridPointLatLon((uint16_t)  cellX,      (uint16_t)  cellY     ); //Upper-left  grid point
			Eigen::Vector2d C2 = GeoRegistrationData->GetGridPointLatLon((uint16_t) (cellX + 1), (uint16_t)  cellY     ); //Upper-right grid point
			Eigen::Vector2d C3 = GeoRegistrationData->GetGridPointLatLon((uint16_t) (cellX + 1), (uint16_t) (cellY + 1)); //Lower-right grid point
			Eigen::Vector2d C4 = GeoRegistrationData->GetGridPointLatLon((uint16_t)  cellX,      (uint16_t) (cellY + 1)); //Lower-left  grid point
			
			Eigen::Vector2d PPrime = P;
			AdjustLongitudesToCommonRange(C1, C2, C3, C4, PPrime); //Map all longitudes to the same side of the discontinuity
			
			double x1 = ((double)  cellX     ) * cellWidth;
			double x2 = ((double) (cellX + 1)) * cellWidth;
			double y1 = ((double)  cellY     ) * cellHeight;
			double y2 = ((double) (cellY + 1)) * cellHeight;
			Eigen::Vector2d candidate = InverseBilinearInterp(PPrime, x1, x2, y1, y2, C1, C2, C3, C4);
			if (! std::isnan(candidate(0))) {
				double d = distFromRectange(candidate, x1, x2, y1, y2);
				if (d < distFromParentCell) {
					distFromParentCell = d;
					v = candidate;
				}
				if (d == 0.0)
					break;
			}
		}
	}
	
	//If we have a valid inverse, make sure that it is inside the image bounds
	if (! std::isnan(v(0))) {
		if ((v(0) < -0.5) || (v(0) > ((double) imageWidth) - 0.5))
			return Eigen::Vector2d(std::nan(""), std::nan(""));
		if ((v(1) < -0.5) || (v(1) > ((double) imageHeight) - 0.5))
			return Eigen::Vector2d(std::nan(""), std::nan(""));
	}
	
	return v;
}

//Get number of custom blocks
uint64_t FRFImage::GetNumberOfCustomBlocks(void) const {
	return ((uint64_t) customBlocks.size());
}

//Returns true if custom block with the given custom block code exists and false otherwise.
bool FRFImage::HasCustomBlock(uint64_t CustomBlockCode) const {
	if (customBlocks.count(CustomBlockCode) > 0U)
		return true;
	else
		return false;
}

//Get const access to a custom block. Returns NULL if there is no custom block with the given custom block code
FRFCustomBlock const * FRFImage::CustomBlock(uint64_t CustomBlockCode) const {
	if (customBlocks.count(CustomBlockCode) > 0U)
		return &(customBlocks.at(CustomBlockCode));
	else
		return NULL;
}

//Get access to a custom block. Returns NULL if there is no custom block with the given custom block code
FRFCustomBlock * FRFImage::CustomBlock(uint64_t CustomBlockCode) {
	if (customBlocks.count(CustomBlockCode) > 0U)
		return &(customBlocks[CustomBlockCode]);
	else
		return NULL;
}

//Add custom block with the given custom block code and return a pointer to the new object
FRFCustomBlock * FRFImage::AddCustomBlock(uint64_t CustomBlockCode) {
	customBlocks[CustomBlockCode] = FRFCustomBlock();
	customBlocks[CustomBlockCode].CustomBlockCode = CustomBlockCode;
	return &(customBlocks[CustomBlockCode]);
}

//Delete a custom block
void FRFImage::RemoveCustomBlock(uint64_t CustomBlockCode) {
	if (customBlocks.count(CustomBlockCode) > 0U)
		customBlocks.erase(CustomBlockCode);
}

//Get a vector of TagCodes for available Camera Info Tags
std::vector<uint16_t> FRFImage::GetAvailableCamInfoTagCodes(void) const {
	std::vector<uint16_t> tagCodes;
	for (auto kv : CameraInformationTags)
		tagCodes.push_back(kv.first);
	return tagCodes;
}

//Convenience function to get human-readable tag name for a given tag
std::string FRFImage::GetCamInfoTagName(uint16_t TagCode) const {
	std::string name;
	switch (TagCode) {
		case  0U: name = std::string("Make");             break;
		case  1U: name = std::string("Model");            break;
		case  2U: name = std::string("BodySerial");       break;
		case  3U: name = std::string("ImagerMake");       break;
		case  4U: name = std::string("ImagerModel");      break;
		case  5U: name = std::string("LensMake");         break;
		case  6U: name = std::string("LensModel");        break;
		case  7U: name = std::string("LensSerial");       break;
		case  8U: name = std::string("FNumber");          break;
		case  9U: name = std::string("FocalLength");      break;
		case 10U: name = std::string("XPitch");           break;
		case 11U: name = std::string("YPitch");           break;
		case 12U: name = std::string("ExposureTime");     break;
		case 13U: name = std::string("CamMatrix");        break;
		case 14U: name = std::string("RadialDistortion"); break;
		case 15U: name = std::string("RadiometricCal");   break;
		case 16U: name = std::string("RawInfo");          break;
		default:  name = std::string("Unrecognized");
	}
	return name;
}

//Check to see if a camera info tag exists
bool FRFImage::hasCamInfoTag(uint16_t TagCode) const {
	if (CameraInformationTags.count(TagCode) > 0U)
		return true;
	else
		return false;
}

//Delete a tag, if it exists
void FRFImage::removeCamInfoTag(uint16_t TagCode) {
	if (CameraInformationTags.count(TagCode) > 0U)
		CameraInformationTags.erase(TagCode);
}

//Retrieve a string-type camera info tag
std::string FRFImage::GetCamInfoTagString(uint16_t TagCode) const {
	//Make sure the provided TagCode actually corresponds to a string-valued tag
	if (TagCode > 7U)
		return std::string();
	
	if (CameraInformationTags.count(TagCode) > 0U) {
		std::vector<uint8_t> const & TagData = CameraInformationTags.at(TagCode);
		auto iter = TagData.cbegin();
		unsigned int maxBytes = (unsigned int) TagData.size();
		return decodeField_String(iter, maxBytes);
	}
	else
		return std::string();
}

//Retrieve a Float64-type camera info tag
double FRFImage::GetCamInfoTagFloat64(uint16_t TagCode) const {
	//Make sure the provided TagCode actually corresponds to a float64-valued tag
	if (! ((TagCode >= 8U) && (TagCode <= 12U))) 
		return std::nan("");
	
	if (CameraInformationTags.count(TagCode) > 0U) {
		std::vector<uint8_t> const & TagData = CameraInformationTags.at(TagCode);
		if (TagData.size() != 8U) {
			fprintf(stderr,"Warning: Incorrect tag length. Dropping.\r\n");
			return std::nan("");
		}
		auto iter = TagData.cbegin();
		return decodeField_float64(iter);
	}
	else
		return std::nan("");
}

//Retrieve a Mat3-type camera info tag
Eigen::Matrix3d FRFImage::GetCamInfoTagMat3(uint16_t TagCode) const {
	//Create the dummy matrix to return if the tag isn't there or if this method is called on a tag with incorrect type.
	Eigen::Matrix3d NaNMat;
	NaNMat << std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan("");
	
	//Make sure the provided TagCode actually corresponds to a Mat3-valued tag
	if (TagCode != 13U)
		return NaNMat;
	
	if (CameraInformationTags.count(TagCode) > 0U) {
		std::vector<uint8_t> const & TagData = CameraInformationTags.at(TagCode);
		if (TagData.size() != 72U) {
			fprintf(stderr,"Warning: Incorrect tag length. Dropping.\r\n");
			return NaNMat;
		}
		auto iter = TagData.cbegin();
		return decodeField_Mat3(iter);
	}
	else
		return NaNMat;
}

//Set a string-type camera info tag
void FRFImage::SetCamInfoTag(uint16_t TagCode, std::string Value) {
	//Make sure the provided TagCode actually corresponds to a string-valued tag
	if (TagCode > 7U) {
		fprintf(stderr,"Warning: Trying to set cam info tag to value with incorrect type. Ignoring.\r\n");
		return;
	}
	
	//Create tag, if needed, and get reference to the tag data
	std::vector<uint8_t> & TagData = CameraInformationTags[TagCode];
	encodeField_String(TagData, Value);
}

//Set a Float64-type camera info tag
void FRFImage::SetCamInfoTag(uint16_t TagCode, double Value) {
	//Make sure the provided TagCode actually corresponds to a float64-valued tag
	if (! ((TagCode >= 8U) && (TagCode <= 12U))) {
		fprintf(stderr,"Warning: Trying to set cam info tag to value with incorrect type. Ignoring.\r\n");
		return;
	}
	
	//Create tag, if needed, and get reference to the tag data
	std::vector<uint8_t> & TagData = CameraInformationTags[TagCode];
	encodeField_float64(TagData, Value);
}

//Set a Mat3-type camera info tag
void FRFImage::SetCamInfoTag(uint16_t TagCode, Eigen::Matrix3d const & Value) {
	//Make sure the provided TagCode actually corresponds to a Mat3-valued tag
	if (TagCode != 13U) {
		fprintf(stderr,"Warning: Trying to set cam info tag to value with incorrect type. Ignoring.\r\n");
		return;
	}
	
	//Create tag, if needed, and get reference to the tag data
	std::vector<uint8_t> & TagData = CameraInformationTags[TagCode];
	encodeField_Mat3(TagData, Value);
}

//Returns radial distortion coefficients in the form <Cx, Cy, a1, a2, a3, a4>
std::tuple<double,double,double,double,double,double> FRFImage::GetCamInfoRadialDistortionCoefficients(void) const {
	if (CameraInformationTags.count(14U) > 0U) {
		std::vector<uint8_t> const & TagData = CameraInformationTags.at(14U);
		if (TagData.size() != 48U) {
			fprintf(stderr,"Warning: Incorrect tag length. Dropping.\r\n");
			return std::make_tuple(std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""));
		}
		auto iter = TagData.cbegin();
		double Cx = decodeField_float64(iter);
		double Cy = decodeField_float64(iter);
		double a1 = decodeField_float64(iter);
		double a2 = decodeField_float64(iter);
		double a3 = decodeField_float64(iter);
		double a4 = decodeField_float64(iter);
		return std::make_tuple(Cx, Cy, a1, a2, a3, a4);
	}
	else
		return std::make_tuple(std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""));
}

//Set radial distortion coefficients. Takes input: <Cx, Cy, a1, a2, a3, a4>
void FRFImage::SetCamInfoRadialDistortionCoefficients(std::tuple<double,double,double,double,double,double> Coefficients) {
	//Create tag, if needed, and get reference to the tag data
	std::vector<uint8_t> & TagData = CameraInformationTags[14U];
	encodeField_float64(TagData, std::get<0>(Coefficients));
	encodeField_float64(TagData, std::get<1>(Coefficients));
	encodeField_float64(TagData, std::get<2>(Coefficients));
	encodeField_float64(TagData, std::get<3>(Coefficients));
	encodeField_float64(TagData, std::get<4>(Coefficients));
	encodeField_float64(TagData, std::get<5>(Coefficients));
}

//Get the value from a given pixel and layer
double FRFImage::GetValue(uint16_t LayerIndex, uint16_t Row, uint16_t Col) const {
	//The layer GetValue method checks to make sure Row and Col are in bounds
	if ((size_t) LayerIndex < Layers.size())
		return Layers[LayerIndex]->GetValue(Row, Col);
	else
		return std::nan("");
}

//Get RGBA triplet, each on scale 0-255, corresponding to the given visualization at the given pixel.
std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> FRFImage::EvaluateVisualization(uint16_t VisualizationIndex, uint16_t Row, uint16_t Col) const {
	if ((size_t) VisualizationIndex >= VisualizationManifest.size())
		return std::make_tuple((uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U);
	FRFVisualization * FRFViz = VisualizationManifest[VisualizationIndex];
	if (FRFViz->isRGB()) {
		FRFVisualizationRGB * viz = dynamic_cast<FRFVisualizationRGB *>(FRFViz);
		double R =  255.0*(GetValue(viz->RedIndex,   Row, Col) - viz->RedMin)  /(viz->RedMax   - viz->RedMin  );
		double G =  255.0*(GetValue(viz->GreenIndex, Row, Col) - viz->GreenMin)/(viz->GreenMax - viz->GreenMin);
		double B =  255.0*(GetValue(viz->BlueIndex,  Row, Col) - viz->BlueMin) /(viz->BlueMax  - viz->BlueMin );
		if (std::isnan(R) || std::isnan(G) || std::isnan(B))
			return std::make_tuple((uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U);
		else {
			R = saturateToRange(R, 0.0, 255.0);
			G = saturateToRange(G, 0.0, 255.0);
			B = saturateToRange(B, 0.0, 255.0);
			return std::make_tuple((uint8_t) R, (uint8_t) G, (uint8_t) B, (uint8_t) 255U);
		}
	}
	else if (FRFViz->isColormap()) {
		FRFVisualizationColormap * viz = dynamic_cast<FRFVisualizationColormap *>(FRFViz);
		double value = GetValue(viz->LayerIndex, Row, Col);
		
		//If the value is invalid, we can stop now and return transparent black
		if (std::isnan(value))
			return std::make_tuple((uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U);
		
		Eigen::Vector3d color; //RGB triple (each component on a scale of 0 to 1)
		if (value <= std::get<0>(viz->SetPoints.front()))
			//Saturate to lowest set point
			color << std::get<1>(viz->SetPoints.front()), std::get<2>(viz->SetPoints.front()), std::get<3>(viz->SetPoints.front());
		else if (value >= std::get<0>(viz->SetPoints.back()))
			//Saturate to highest set point
			color << std::get<1>(viz->SetPoints.back()), std::get<2>(viz->SetPoints.back()), std::get<3>(viz->SetPoints.back());
		else {
			//We need to identify which set points we lie between and interpolate the color between them
			uint32_t P2Index = 0U;
			for (; (P2Index + 1U) < (uint32_t) viz->SetPoints.size(); P2Index++) {
				if (std::get<0>(viz->SetPoints[P2Index]) > value)
					break;
			}
			uint32_t P1Index = P2Index > 0U ? P2Index - 1U : 0U;
			
			double P1Value = std::get<0>(viz->SetPoints[P1Index]);
			double P2Value = std::get<0>(viz->SetPoints[P2Index]);
			if (P2Value - P1Value <= 0.0)
				//If for some reason the set points are the same or share the same value (not technically allowed), return the color of the first set point
				color << std::get<1>(viz->SetPoints[P1Index]), std::get<2>(viz->SetPoints[P1Index]), std::get<3>(viz->SetPoints[P1Index]);
			else {
				//Interpolate the color between set points P1 and P2
				Eigen::Vector3d P1Color(std::get<1>(viz->SetPoints[P1Index]), std::get<2>(viz->SetPoints[P1Index]), std::get<3>(viz->SetPoints[P1Index]));
				Eigen::Vector3d P2Color(std::get<1>(viz->SetPoints[P2Index]), std::get<2>(viz->SetPoints[P2Index]), std::get<3>(viz->SetPoints[P2Index]));
				double t1 = (P2Value - value) / (P2Value - P1Value);
				double t2 = (value - P1Value) / (P2Value - P1Value);
				color = t1*P1Color + t2*P2Color;
			}
		}
		
		//Re-scale colors to be between 0 and 255 and saturate to these endpoints in case of round-off error or invalid set points
		color(0) = saturateToRange(255.0*color(0), 0.0, 255.0);
		color(1) = saturateToRange(255.0*color(1), 0.0, 255.0);
		color(2) = saturateToRange(255.0*color(2), 0.0, 255.0);
		
		return std::make_tuple((uint8_t) color(0), (uint8_t) color(1), (uint8_t) color(2), (uint8_t) 255U);
	}
	else {
		fprintf(stderr,"Warning: Unrecognized visualization type.\r\n");
		return std::make_tuple((uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U, (uint8_t) 0U);
	}
}

//Evaluate a visualization on a sequence of pixels and write the results as packed RGBA tripples (each pixel takes 32 consecutive bits) into the target buffer
void FRFImage::EvaluateVisualizationForRow(uint16_t VisualizationIndex, uint16_t Row, uint16_t StartCol, uint16_t PixelCount, uint8_t * TargetBuffer) const {
	fprintf(stderr,"Warning: Row-based viz evaluation not accelerated yet.\r\n");
	for (uint16_t pixelCounter = 0U; pixelCounter < PixelCount; pixelCounter++) {
		std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color = EvaluateVisualization(VisualizationIndex, Row, StartCol + pixelCounter);
		TargetBuffer[4U * ((uint32_t) pixelCounter)     ] = std::get<0>(color);
		TargetBuffer[4U * ((uint32_t) pixelCounter) + 1U] = std::get<1>(color);
		TargetBuffer[4U * ((uint32_t) pixelCounter) + 2U] = std::get<2>(color);
		TargetBuffer[4U * ((uint32_t) pixelCounter) + 3U] = std::get<3>(color);
	}
}

// *******************************************************************************************************************************
// ****************************************   Load and Inspect Example Shadow Map File   *****************************************
// *******************************************************************************************************************************
static void printLayerInfo(FRFLayer* Layer) {
	fprintf(stderr, "Name: %s\r\n", Layer->Name.c_str());
	fprintf(stderr, "Description: %s\r\n", Layer->Description.c_str());
	fprintf(stderr, "Units: %s\r\n", Layer->GetUnitsStringInMostReadableForm().c_str());
	fprintf(stderr, "Type Code: %u (%s)\r\n", (unsigned int)Layer->GetTypeCode(), Layer->GetTypeString().c_str());
	fprintf(stderr, "alpha: %.8e,  beta: %.8f\r\n", Layer->alpha, Layer->beta);
	fprintf(stderr, "Validity Mask: %s\r\n", Layer->HasValidityMask ? "Yes" : "No");
	fprintf(stderr, "Total size: %f MB\r\n", ((double)(Layer->GetTotalBytesWithValidityMask() / ((uint64_t)100000U))) / 10.0);

	std::tuple<double, double> range = Layer->GetRange();
	if (Layer->GetTypeCode() <= 68U) {
		fprintf(stderr, "Value Range: [%f, %f]\r\n", std::get<0>(range), std::get<1>(range));
		fprintf(stderr, "Value Step Size: %e\r\n", Layer->GetStepSize());
	}
	else
		fprintf(stderr, "Value Range: [%e, %e]\r\n", std::get<0>(range), std::get<1>(range));
}

static void LoadImageAndInspectShadowMapFile(std::string Filename) {
	FRFImage shadowMap; //Create a new empty FRF Image

	//Clear the image object and load from disk
	if (shadowMap.LoadFromDisk(Filename))
		std::cerr << "FRF Image loaded successfully.\r\n";
	else
		std::cerr << "FRF Image loading failed.\r\n";

	//Check to see if the FRF file is actually a shadow map... it should have a shadow map information custom block
	if (IsShadowMapFile(shadowMap))
		std::cerr << "This is a shadow map.\r\n";
	else {
		std::cerr << "This is not a shadow map! Stopping now.\r\n";
		return;
	}

	//Print out some basic info about the file
	std::tuple<uint16_t, uint16_t> fileVersion = shadowMap.getFileVersion();
	fprintf(stderr, "FRF version %u.%u\r\n", (unsigned int)std::get<0>(fileVersion), (unsigned int)std::get<1>(fileVersion));
	fprintf(stderr, "Dimensions: %u rows x %u cols\r\n", (unsigned int)shadowMap.Height(), (unsigned int)shadowMap.Width());
	fprintf(stderr, "Number of Layers: %u,  Number of Visualizations: %u\r\n", (unsigned int)shadowMap.NumberOfLayers(),
		(unsigned int)shadowMap.GetNumberOfVisualizations());
	fprintf(stderr, "Shadow Map is Geo-Registered: %s\r\n", shadowMap.IsGeoRegistered() ? "Yes" : "No");
	if (shadowMap.IsGeoRegistered()) {
		std::tuple<double, double> LatLon = shadowMap.GetCoordinatesOfPixel((uint16_t)0U, (uint16_t)7U);
		std::cerr << "The latitude  of pixel (row, col) = (0, 7) is: " << std::get<0>(LatLon) * 180.0 / PI << " degrees.\r\n";
		std::cerr << "The longitude of pixel (row, col) = (0, 7) is: " << std::get<1>(LatLon) * 180.0 / PI << " degrees.\r\n";
	}
	std::cerr << "\r\n";

	//Print out some layer info for each layer
	for (uint16_t layerIndex = 0U; layerIndex < shadowMap.NumberOfLayers(); layerIndex++) {
		fprintf(stderr, "\r\n**************  Layer %u  **************\r\n", (unsigned int)layerIndex);
		printLayerInfo(shadowMap.Layer(layerIndex));
	}
	std::cerr << "\r\n";

	//Print out some pixel values from the first layer
	if (shadowMap.NumberOfLayers() == 0U)
		std::cerr << "Uh-oh. This image is empty. Nothing to display.\r\n";
	else {
		std::cerr << "Value at (73,24): " << shadowMap.Layer(0U)->GetValue(73U, 24U) << "\r\n";
		std::cerr << "Value at (12,82): " << shadowMap.Layer(0U)->GetValue(12U, 82U) << "\r\n";
		std::cerr << "Value at (31,3900): " << shadowMap.Layer(0U)->GetValue(31U, 3900U) << "\r\n";
		std::cerr << "Value at (19,122): " << shadowMap.Layer(0U)->GetValue(19U, 122U) << "\r\n";
		std::cerr << "Value at (19,123): " << shadowMap.Layer(0U)->GetValue(19U, 123U) << "\r\n";
		std::cerr << "Value at (19,124): " << shadowMap.Layer(0U)->GetValue(19U, 124U) << "\r\n";
	}
	std::cerr << "\r\n";

	//Inspect the shadow map information block
	ShadowMapInfoBlock myShadowMapInfoBlock;
	if (myShadowMapInfoBlock.LoadFromFRFFile(shadowMap)) {
		if (std::isnan(myShadowMapInfoBlock.FileTimeEpoch_TOW))
			std::cerr << "Absolute time information not available.\r\n";
		else {
			std::cerr << "Absolute time information is known.\r\n";
			std::cerr << "File 0-time epoch corresponds with GPS Week " << myShadowMapInfoBlock.FileTimeEpoch_Week <<
				", TOW = " << myShadowMapInfoBlock.FileTimeEpoch_TOW << " seconds.\r\n";
		}
		for (uint16_t layerIndex = 0U; layerIndex < shadowMap.NumberOfLayers(); layerIndex++)
			std::cerr << "Layer " << layerIndex << " file-time timestamp: " << myShadowMapInfoBlock.LayerTimeTags[layerIndex] << " seconds.\r\n";
		std::cerr << "\r\n";
	}
	else
		std::cerr << "Error: shadow map information block could not be decoded. Corrupt or non-conformant file.\r\n";
}


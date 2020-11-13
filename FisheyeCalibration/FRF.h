#pragma once

//System Includes
#include <stdint.h>
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>

// ****************************************************************************************************************************************
// ********************************************************   Component Classes   *********************************************************
// ****************************************************************************************************************************************
class FRFLayer {
	public:
		FRFLayer() = delete;
		FRFLayer(uint16_t Width, uint16_t Height); //Defaults to uint8 layer with no validity mask, alpha=1, beta=0, unspecified units, and blank Name/Description.
		
		std::string Name;
		std::string Description;
		int32_t UnitsCode;
		double alpha;
		double beta;
		bool HasValidityMask;
		std::vector<uint8_t> Data; //Raw data for layer - this can be read and modified directly for sake of speed if you know what you are doing
		
		uint8_t GetTypeCode(void) const;
		void    SetTypeCode(uint8_t NewTypeCode); //After changing the type, you must re-allocate the storage before accessing values
		
		//This helper sets alpha and beta so that the lower end of the raw value range corresponds to MinVal and the upper end corresponds to MaxVal
		//If you change the data type for the layer you will to set alpha and beta again to get the desired range.
		void SetAlphaAndBetaForGivenRange(double MinVal, double MaxVal);
		
		std::string GetTypeString(void) const; //Get human-readable string indicating layer data type (e.g. 8-bit uint, 32-bit float)
		std::tuple<double, double> GetRange(void) const; //Get lowest and highest values representable by layer, as configured. Returned as <lower, upper>
		double GetStepSize(void) const;  //Gets the difference between consecutive representable values
		double GetTolerance(void) const; //Gets slightly higher than half the step size. This is a tight upper bound on the error introduced by our numerical encoding.
		
		uint64_t GetTotalBytesWithoutValidityMask(void) const;
		uint64_t GetTotalBytesWithValidityMask(void) const;
		
		void AllocateStorage(void);   //Allocate memory for the image data and (if applicable) the validity mask
		void DeAllocateStorage(void); //Free memory for the image data and validity mask (setting values will fail and retrieval will return NaN if not allocated)
		
		//Get or set the value of a pixel. These use alpha and beta to correctly interpret stored data and encode values in the correct range.
		//An invalid pixel will return NaN. Set a pixel to NaN to mark it as invalid
		double GetValue(uint32_t Row, uint32_t Col) const;
		bool   SetValue(uint32_t Row, uint32_t Col, double Value);
		
		std::string GetUnitsStringInExponentForm(void);     //Get a string showing units as a product of the SI base units with exponents, such as (kg m^2 s^-3)
		std::string GetUnitsStringInMostReadableForm(void); //Get a string showing units in the most human-readable form
	
	private:
		uint16_t layerWidth;  //Right now, this must match the Image Width in the owning FRFImage
		uint16_t layerHeight; //Right now, this must match the Image Height in the owning FRFImage
		
		//We update the bitsPerSample field whenever TypeCode changes, so it is not publicly accessable
		uint8_t TypeCode;
		uint32_t bitsPerSample;
		void updateBitsPerSample(void);
		
		//Helper function for range, tolerance, and coefficient-setting methods
		std::tuple<double, double> GetRawValueLimits(void) const;
		
		//Helper functions for accessing a validity mask. Only use for non-float layers (validity masks are not allowed for float layers)
		bool GetValidityFromMask(uint32_t Row, uint32_t Col) const;     //Check pixel validity (returns true if there is no mask)
		void SetValidityInMask(uint32_t Row, uint32_t Col, bool Valid); //Mark a pixel as valid or invalid in mask. Does nothing if there is no mask.
		
		//Different versions of value getters and setters
		double GetValue_Generic(uint32_t Row, uint32_t Col) const;
		bool SetValue_Generic(uint32_t Row, uint32_t Col, double Value);
};

//Abstract class for visualizations
class FRFVisualization {
	public:
		std::string Name;
		std::string Description;
		
		//Virtual destructor
		virtual ~FRFVisualization() = default;
		
		//Inspection
		virtual bool isRGB(void) const = 0;
		virtual bool isColormap(void) const = 0;
		
		//Object cloning (Virtual Copy Constructor)
		virtual FRFVisualization * Clone() const = 0;
};

class FRFVisualizationRGB : public FRFVisualization {
	public:
		uint16_t RedIndex;
		double   RedMin;
		double   RedMax;
		uint16_t GreenIndex;
		double   GreenMin;
		double   GreenMax;
		uint16_t BlueIndex;
		double   BlueMin;
		double   BlueMax;
		
		//Inspection
		bool isRGB(void) const override;
		bool isColormap(void) const override;
		
		//Create a copy of this object and return a pointer to it of the base class type
		FRFVisualization * Clone() const override { return new FRFVisualizationRGB(*this); }
};

class FRFVisualizationColormap : public FRFVisualization {
	public:
		uint16_t LayerIndex;
		
		//The SetPoints muct be sorted in increasing order of Value. They will be sorted automatically when reading from disk, but if you create
		//a colormap visualization, you must make sure the ordering rule is obeyed before using or saving the visualization. A sort method is provided for convenience.
		std::vector<std::tuple<double,double,double,double>> SetPoints; //Each item has form <Value, Red, Green, Blue>
		
		//Inspection
		bool isRGB(void) const override;
		bool isColormap(void) const override;
		
		//This is a convenience method for sorting set points in increasing order of value. This is only needed if you create an object manually.
		void SortSetPoints(void);
		
		//Create a copy of this object and return a pointer to it of the base class type
		FRFVisualization * Clone() const override { return new FRFVisualizationColormap(*this); }
};

class FRFGeoTag {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
		Eigen::Vector3d P_ECEF;     //All elements set to NaN if unknown or unspecified
		Eigen::Matrix3d C_ECEF_Cam; //All elements set to NaN if unknown or unspecified
		uint32_t        GPST_Week;  //Set to 0U if unknown or unspecified
		double          GPST_TOW;   //Set to NaN if unknown or unspecified
};

class FRFGeoRegistration {
	public:
		uint16_t RegistrationType;
		double Altitude;
		uint16_t GridWidthDivisor;
		uint16_t GridHeightDivisor;
		std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> GridPointLatLons; //<Lat,Lon> pairs (in radians) for each grid point (sorted row by row)
		
		Eigen::Vector2d GetGridPointLatLon(uint16_t PointX, uint16_t PointY) const; //Retrieve the Lat and Lon (radians) of a grid point from the grid location
		void RegisterFromCornerLocations(Eigen::Vector2d const & UL, Eigen::Vector2d const & UR, Eigen::Vector2d const & LL, Eigen::Vector2d const & LR); //Populate all fields except Altitude based on image corner locations
		
};

class FRFCustomBlock {
	public:
		uint64_t CustomBlockCode;
		std::vector<uint8_t> BlockPayload;
};


// ****************************************************************************************************************************************
// *********************************************************   FRF Image Class   **********************************************************
// ****************************************************************************************************************************************
class FRFImage {
	public:
		FRFImage(); //Create an empty FRF image
		FRFImage(std::string Filepath); //Try to load an FRF image from disk. On failure, create an empty image
		FRFImage(const FRFImage & Other); //Copying images is allowed, but each copy is deep and therefore expensive for large images, so use with caution
		~FRFImage();
		
		//Load and Save methods
		bool LoadFromDisk(std::string Filepath); //Try to load an FRF image from disk. On failure, create an empty image
		bool SaveToDisk(std::string Filepath) const;
		
		//General Inspection and Modification Methods
		std::tuple<uint16_t, uint16_t> getFileVersion(void) const; //Return <Major, Minor> version tuple for the image
		void     Clear(void);                 //Destroy image and reset object to default-initialized state
		uint16_t Width(void) const;           //Get image width (pixels)
		uint16_t Height(void) const;          //Get image height (pixels)
		uint16_t Cols(void) const;            //Same as Width()
		uint16_t Rows(void) const;            //Same as Height()
		bool     SetWidth(uint16_t Width);    //Set image width (pixels). Can only be performed on an empty image (no layers).
		bool     SetHeight(uint16_t Height);  //Set image height (pixels). Can only be performed on an empty image (no layers).
		
		//Layer access
		uint16_t NumberOfLayers(void) const;
		FRFLayer const * Layer(uint16_t LayerIndex) const; //Get const access to a layer for reading
		FRFLayer       * Layer(uint16_t LayerIndex);       //Get access to a layer with permission to modify
		FRFLayer       * AddLayer(void);                   //Add a new layer to the image and return a pointer to the new layer (with permission to modify)
		void RemoveLayer(uint16_t LayerIndex);
		bool     HasAlphaLayer(void) const;                //Returns true if the image has an alpha layer and false otherwise
		uint16_t GetAlphaLayerIndex(void) const;           //Get the index of the alpha layer (returns 2^16-1 if there is no alpha layer)
		void     SetAlphaLayerIndex(uint16_t Index);       //Set the alpha layer index
		void     RemoveAlphaLayer(void);                   //Clear the alpha layer designation (this does not destroy any layers - it just unmarks the alpha layer, if there is one)
		
		//Visualization Access
		uint16_t GetNumberOfVisualizations(void) const;
		FRFVisualization const   * Visualization(uint16_t VisualizationIndex) const; //Get const access to a layer for reading
		FRFVisualization         * Visualization(uint16_t VisualizationIndex);       //Get access to a layer with permission to modify
		FRFVisualizationRGB      * AddVisualizationRGB(void);                        //Add a new RGB visualization and return pointer to it (with permission to modify)
		FRFVisualizationColormap * AddVisualizationColormap(void);                   //Add a new Colormap visualization and return pointer to it (with permission to modify)
		void RemoveVisualization(uint16_t VisualizationIndex);
		
		//GeoTag Access
		bool HasGeoTag(void) const;              //Returns true if the object has a GeoTag and false otherwise
		FRFGeoTag const * GetGeoTag(void) const; //Get const pointer to GeoTag. Returns NULL if there is no GeoTag
		FRFGeoTag       * GetGeoTag(void);       //Get regular pointer to GeoTag. Returns NULL if there is no GeoTag
		void SetGeoTag(FRFGeoTag GeoTag);        //Sets the GeoTag
		void RemoveGeoTag(void);                 //Get rid of GeoTag, if there is one
		
		//GeoRegistration Access
		bool IsGeoRegistered(void) const;                            //Returns true if the object has GeoRegistration data and false otherwise
		FRFGeoRegistration const * GetGeoRegistration(void) const;   //Get const access to GeoRegistration data. Returns NULL if non-existent
		FRFGeoRegistration       * GetGeoRegistration(void);         //Get access to GeoRegistration data. Returns NULL if non-existent
		void SetGeoRegistration(FRFGeoRegistration GeoRegistration); //Set the GeoRegistration data
		void RemoveGeoRegistration(void);                            //Get rid of the GeoRegistration data, if there was any
		std::tuple<double, double> GetCoordinatesOfPixel(uint16_t Row, uint16_t Col) const;          //Returns <Lat, Lon> tuple (both in radians) for the center of a specific pixel
		std::tuple<double, double> GetCoordinatesOfPixel(Eigen::Vector2d const & PixelCoords) const; //Returns <Lat, Lon> tuple (both in radians) for arbitrary (x,y) pixel coordinates
		Eigen::Vector2d GetPixelCoordsForLocation(double Lat, double Lon) const;                     //Returns (x,y) pixel coordinates corresponding to a given Lat, Lon location.
		
		//Custom Block Access
		uint64_t GetNumberOfCustomBlocks(void) const;
		bool HasCustomBlock(uint64_t CustomBlockCode) const;                //Returns true if custom block with the given custom block code exists and false otherwise.
		FRFCustomBlock const * CustomBlock(uint64_t CustomBlockCode) const; //Get const access to a custom block. Returns NULL if there is no custom block with the given custom block code
		FRFCustomBlock       * CustomBlock(uint64_t CustomBlockCode);       //Get access to a custom block. Returns NULL if there is no custom block with the given custom block code
		FRFCustomBlock       * AddCustomBlock(uint64_t CustomBlockCode);    //Add custom block with the given custom block code and return a pointer to the new object
		void RemoveCustomBlock(uint64_t CustomBlockCode);
		
		//Camera Information Tag Access - Tags have values of varying types, which is why there are multiple versions of the accessors. Use the appropriate accessor for
		//the tag you are setting. Using the wrong "getter" will return a default value (like an empty string or 0.0). Using the wrong "setter" will result in no action.
		std::vector<uint16_t> GetAvailableCamInfoTagCodes(void) const; //Get a vector of TagCodes for available Camera Info Tags
		std::string     GetCamInfoTagName(uint16_t TagCode) const;     //Convenience function to get human-readable tag name for a given tag
		bool            hasCamInfoTag(uint16_t TagCode) const;
		void            removeCamInfoTag(uint16_t TagCode);
		std::string     GetCamInfoTagString(uint16_t TagCode) const;
		double          GetCamInfoTagFloat64(uint16_t TagCode) const;
		Eigen::Matrix3d GetCamInfoTagMat3(uint16_t TagCode) const;
		void            SetCamInfoTag(uint16_t TagCode, std::string Value);
		void            SetCamInfoTag(uint16_t TagCode, double Value);
		void            SetCamInfoTag(uint16_t TagCode, Eigen::Matrix3d const & Value);
		
		//We have a few special Camera Info Tags that have unique types. These have special getters and setters
		std::tuple<double,double,double,double,double,double> GetCamInfoRadialDistortionCoefficients(void) const; //Returns <Cx, Cy, a1, a2, a3, a4>
		void SetCamInfoRadialDistortionCoefficients(std::tuple<double,double,double,double,double,double> Coefficients); //Takes input: <Cx, Cy, a1, a2, a3, a4>
		
		//Utility methods for getting access to imagery and "evaluating" visualizations
		double GetValue(uint16_t LayerIndex, uint16_t Row, uint16_t Col) const;
		std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> EvaluateVisualization(uint16_t VisualizationIndex, uint16_t Row, uint16_t Col) const; //Returns RGBA triplet, each on scale 0-255
		void EvaluateVisualizationForRow(uint16_t VisualizationIndex, uint16_t Row, uint16_t StartCol, uint16_t PixelCount, uint8_t * TargetBuffer) const;
		
	private:
		uint16_t majorVersion;
		uint16_t minorVersion;
		uint16_t imageWidth;
		uint16_t imageHeight;
		uint16_t alphaLayerIndex; //Set to 2^16-1 if there is no alpha layer
		
		std::vector<FRFLayer *> Layers;
		std::vector<FRFVisualization *> VisualizationManifest;
		
		FRFGeoTag * GeoTagData = NULL; //Null if not present/unused
		FRFGeoRegistration * GeoRegistrationData = NULL; //Null if not present/unused
		
		std::unordered_map<uint16_t, std::vector<uint8_t>> CameraInformationTags; //Indexed by TagCode
		std::unordered_map<uint64_t, FRFCustomBlock> customBlocks; //Indexed by Custom Block Code
};



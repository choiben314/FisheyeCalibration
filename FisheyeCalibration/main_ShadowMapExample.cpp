//System Includes
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <bitset>
#include <chrono>
#include <random>

//Project Includes
#include "FRF.h"
#include "ShadowMapIO.hpp"

#define PI 3.14159265358979

// *******************************************************************************************************************************
// *********************************************   Create Example Shadow Map File   **********************************************
// *******************************************************************************************************************************
static void CreateSampleShadowMapFile(std::string Filename) {
	FRFImage shadowMap; //Create new FRF file
	
	//Set Image dimensions - this must be done now, when there are no layers in the image yet.
	if (! shadowMap.SetWidth(512U))
		std::cerr << "Failed to set image width.\r\n";
	if (! shadowMap.SetHeight(512U))
		std::cerr << "Failed to set image height.\r\n";
	
	//Add 3 layers to the image. We can have as few as 1 and as many as 2048.
	for (int layerNum = 0; layerNum < 3; layerNum++) {
		//Add a new layer and set it up
		FRFLayer * newLayer = shadowMap.AddLayer();
		newLayer->Name = std::string("Shadow Map Layer");
		newLayer->Description = std::string("0 = Unshadowed, 1 = Fully shadowed");
		newLayer->UnitsCode = -1; //No units
		newLayer->SetTypeCode(8U); //See Table 1 in the spec. We are going to use 8-bit unsigned integers for each pixel in this layer
		newLayer->HasValidityMask = true; //Add validity info for each pixel in this layer
		newLayer->SetAlphaAndBetaForGivenRange(0.0, 1.0); //Let the FRF lib set coefficients so values are in range [0,1]
		newLayer->AllocateStorage(); //This needs to be called before the layer can be accessed
		
		//Use a nested loop like this to fill in the layer from whatever RAM-based container you are using (like an OpenCV Mat)
		//Set a pixel value to NaN to mark the pixel as invalid - this works even though we are using Uint8 as the underlying type for the layer.
		//For valid pixels you should set the value to something between 0 and 1. Your value will be rounded to the closest achievable value using the
		//bit depth for the layer. Since we are using 8 bits, there are 256 achievable values in the range [0,1]. If we used 1 bit instead, then the only
		//achievable values are 0, 1, and NaN. Any numeric value would result in 0 or 1... whichever is closer.
		for (uint32_t row = 0U; row < shadowMap.Rows(); row++) {
			for (uint32_t col = 0U; col < shadowMap.Cols(); col++)
				newLayer->SetValue(row, col, std::nan(""));
		}
	}
	
	//Set some values in layer 0 to non-trivial values just as an example
	shadowMap.Layer(0)->SetValue(73U,   24U, 0.0);
	shadowMap.Layer(0)->SetValue(12U,   82U, 0.3); //Will round to the closest value achievable using the specified bit depth of the layer
	shadowMap.Layer(0)->SetValue(31U, 3900U, 1.0); //Writing out of bounds is safe, but won't do anything.
	shadowMap.Layer(0)->SetValue(19U,  122U, 2.7); //Will saturate to 1.0... the max value we set for the layer
	shadowMap.Layer(0)->SetValue(19U,  123U, 1.0);
	
	//FRF requires at least one visualization, which can be used to display the imagery visually by a viewer. This isn't super important for us but FRF requires
	//it so everybody "sees" the same thing when an image is viewed. We will make a visualization that maps unshadowed to white and fully shadowed to black. All
	//values in between 0 and 1 will get mapped implicitly to shades of gray.
	FRFVisualizationColormap * viz = shadowMap.AddVisualizationColormap();
	viz->LayerIndex = 0U; //Base the visualization on the first layer of the shadow map
	viz->SetPoints.push_back(std::make_tuple(0.0, 1.0, 1.0, 1.0)); //Map value 0 (unshadowed) to white (RGB all set to 1)
	viz->SetPoints.push_back(std::make_tuple(1.0, 0.0, 0.0, 0.0)); //Map value 1 (fully shadowed) to black (RGB all set to 0)
	
	//Make a Geo-Registration tag. This is only possible if the imagery is registered and it is not strictly required to create a valid shadow map file.
	//We will provide the GPS coordinates of the 4 corners of the image and let the FRF library fill in the block for us.
	FRFGeoRegistration GeoRegistrationTag;
	GeoRegistrationTag.Altitude = std::nan(""); //Raster layer is associated with the Earth's surface
	Eigen::Vector2d UL(43.565718*(PI/180.0), -92.063605*(PI/180.0)); //(Latitude, Longitude) pair (in radians) of the center of the upper-left pixel of the image
	Eigen::Vector2d UR(43.565718*(PI/180.0), -92.056537*(PI/180.0)); //(Latitude, Longitude) pair (in radians) of the center of the upper-right pixel of the image
	Eigen::Vector2d LL(43.561301*(PI/180.0), -92.063605*(PI/180.0)); //(Latitude, Longitude) pair (in radians) of the center of the lower-left pixel of the image
	Eigen::Vector2d LR(43.561301*(PI/180.0), -92.056537*(PI/180.0)); //(Latitude, Longitude) pair (in radians) of the center of the lower-right pixel of the image
	GeoRegistrationTag.RegisterFromCornerLocations(UL, UR, LL, LR);
	shadowMap.SetGeoRegistration(GeoRegistrationTag);
	
	ShadowMapInfoBlock myShadowMapInfoBlock;
	myShadowMapInfoBlock.FileTimeEpoch_Week = 0U;
	myShadowMapInfoBlock.FileTimeEpoch_TOW  = std::nan("");
	myShadowMapInfoBlock.LayerTimeTags.push_back(0.0); //Set the time tag for layer 0 to 0.0 seconds
	myShadowMapInfoBlock.LayerTimeTags.push_back(1.0); //Set the time tag for layer 1 to 1.0 seconds
	myShadowMapInfoBlock.LayerTimeTags.push_back(2.0); //Set the time tag for layer 2 to 2.0 seconds
	if (! myShadowMapInfoBlock.AttachToFRFFile(shadowMap))
		std::cerr << "Error adding shadow map info block... do we have the right number of time tags? There should be 1 per layer.\r\n";
	
	//Save the shadow map file to disk
	shadowMap.SaveToDisk(Filename);
}


// *******************************************************************************************************************************
// ****************************************   Load and Inspect Example Shadow Map File   *****************************************
// *******************************************************************************************************************************
static void printLayerInfo(FRFLayer * Layer) {
	fprintf(stderr, "Name: %s\r\n", Layer->Name.c_str());
	fprintf(stderr, "Description: %s\r\n", Layer->Description.c_str());
	fprintf(stderr, "Units: %s\r\n", Layer->GetUnitsStringInMostReadableForm().c_str());
	fprintf(stderr, "Type Code: %u (%s)\r\n", (unsigned int) Layer->GetTypeCode(), Layer->GetTypeString().c_str());
	fprintf(stderr, "alpha: %.8e,  beta: %.8f\r\n", Layer->alpha, Layer->beta);
	fprintf(stderr, "Validity Mask: %s\r\n", Layer->HasValidityMask ? "Yes" : "No");
	fprintf(stderr, "Total size: %f MB\r\n", ((double) (Layer->GetTotalBytesWithValidityMask() / ((uint64_t) 100000U)))/10.0);
	
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
	fprintf(stderr,"FRF version %u.%u\r\n", (unsigned int) std::get<0>(fileVersion), (unsigned int) std::get<1>(fileVersion));
	fprintf(stderr,"Dimensions: %u rows x %u cols\r\n", (unsigned int) shadowMap.Height(), (unsigned int) shadowMap.Width());
	fprintf(stderr,"Number of Layers: %u,  Number of Visualizations: %u\r\n", (unsigned int) shadowMap.NumberOfLayers(),
	                                                                          (unsigned int) shadowMap.GetNumberOfVisualizations());
	fprintf(stderr,"Shadow Map is Geo-Registered: %s\r\n", shadowMap.IsGeoRegistered() ? "Yes" : "No");
	if (shadowMap.IsGeoRegistered()) {
		std::tuple<double, double> LatLon = shadowMap.GetCoordinatesOfPixel((uint16_t) 0U, (uint16_t) 7U);
		std::cerr << "The latitude  of pixel (row, col) = (0, 7) is: " << std::get<0>(LatLon)*180.0/PI << " degrees.\r\n";
		std::cerr << "The longitude of pixel (row, col) = (0, 7) is: " << std::get<1>(LatLon)*180.0/PI << " degrees.\r\n";
	}
	std::cerr << "\r\n";
	
	//Print out some layer info for each layer
	for (uint16_t layerIndex = 0U; layerIndex < shadowMap.NumberOfLayers(); layerIndex++) {
		fprintf(stderr,"\r\n**************  Layer %u  **************\r\n", (unsigned int) layerIndex);
		printLayerInfo(shadowMap.Layer(layerIndex));
	}
	std::cerr << "\r\n";
	
	//Print out some pixel values from the first layer
	if (shadowMap.NumberOfLayers() == 0U)
		std::cerr << "Uh-oh. This image is empty. Nothing to display.\r\n";
	else {
		std::cerr << "Value at (73,24): "   << shadowMap.Layer(0U)->GetValue(73U, 24U)   << "\r\n";
		std::cerr << "Value at (12,82): "   << shadowMap.Layer(0U)->GetValue(12U, 82U)   << "\r\n";
		std::cerr << "Value at (31,3900): " << shadowMap.Layer(0U)->GetValue(31U, 3900U) << "\r\n";
		std::cerr << "Value at (19,122): "  << shadowMap.Layer(0U)->GetValue(19U, 122U)  << "\r\n";
		std::cerr << "Value at (19,123): "  << shadowMap.Layer(0U)->GetValue(19U, 123U)  << "\r\n";
		std::cerr << "Value at (19,124): "  << shadowMap.Layer(0U)->GetValue(19U, 124U)  << "\r\n";
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

int main(int argc, char* argv[]) {
	std::string Filename("MyShadowMap.frf");
	
	CreateSampleShadowMapFile(Filename);
	
	LoadImageAndInspectShadowMapFile(Filename);
	
	return 0;
}







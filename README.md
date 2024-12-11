# Image Generator Tester Notebook

Purpose of the notebook is to test the image generator. The generator is supposed to generate images from an input dataset. The dataset must be in the input folder following the structure:

```
    input_path
        |--- class1
        |--- class2
        |--- ...
        |--- classN
```

The generator will produce an output folder with the following structure:

```
    output_path
        |--- sampled_images.zip
        |--- input_mapping.parquet
        |--- target.parquet
```

Whereas, the sampled_images.zip might be structured as follows, according to the `organize_by_class` parameter:

- `organize_by_class=False`: (Default) folder containing all the images, without a class folder structure
- `organize_by_class=True`: folder containing all the images, organized by class, following the input folder structure.

### How does the generator work?

The generator will sample images from the input folder and store generated ones into a zip file. An example of workflow is the following:

1. Instantiate the generator with the input and output paths
2. Call the `add_abrupt_drift` method to instanciate a transformation pipeline. This method supports two modes:
   - `drift_lefel`: given a float value of a drift level, between 0 and 1, the method will instantiate a defaul transformation pipeline using Gaussian Blur and Gaussian Noise transformation.
   - `transform_list`: given a list of transformations, the method will instantiate a transformation pipeline, relying on the availabe transformations listed in the class.
3. Call the `sample` method to generate the images and store them in the output folder along with the input mapping and target parquet files.

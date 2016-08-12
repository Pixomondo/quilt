Copyright 2016 Pixomondo LLC.


Quilt is a tool to create a bigger texture from a small one. Instead of 
patching the input texture and then manually removing seams and 
repetitions, this tool automatically breaks the texture into small tiles
and seamlessly recombines them in the output image; doing so the 
original texture structure is preserved and repetitions hard to notice.

![Alt text](data/figures/show_case.jpg?raw=true "Show case")


Creation Process
----------------
The output texture is created composing overlapping tiles from the input 
texture in the following way:
Given the already-placed tile B1, we choose the tile B2 as the best 
matching tile (or a random choice among the best matching tiles), that 
is the tile with most similar overlapping area. 
Once found, we cut the two tiles on the pixels that match the best. To 
do that we compute the minimal cut in the overlapping area and cut the 
tiles accordingly. In this way no seam will be visible.

![Alt text](data/figures/quilt_schema.png?raw=true "Tile Schema")

**Patch matching**
If a patch has already been placed in the output image, the adjacent 
patch is found in the following way:
 1. consider the overlap area of the already placed patch 
 2. compute the distance between this overlapping area and the source 
image. The distance is computed through the sum of squared differences
sum( (img-overlap)^2 ) = sum(img^2) + sum(overlap^2) - 2\*img\*overlap
which in this implementation is expressed through Einstein summation 
(see ssd.py)
 3. consider the minima of the so-found distance matrix, that is the 
 values < (minimum + error). In this way, the amount of error controls 
 how accurate is the choice of the new patch (whose overlap area is 
 very similar to the already placed one), versus how much variety is 
 accepted.


Getting Started
---------------
Quilt is a command line tool. The only required argument is the path to 
the input texture. E.g.:
```
quilt C:/data/image.jpg
```

Basic parameters:

- tilesize: size of the tiles, i.e. units of the texture synthesis 
process (default: 30). 
- overlap: amount of overlap between two tiles (default: 10). Note: the
bigger the overlap is the longer is the computation, but the less seams
will be visible in the texture. Overlap value should be > 0, 
< (tilesize/2).
- error: amount of error accepted in the selection of the tiles. A small
error (e.g. 0.002) reduces the probability of artifacts, but increase 
the one of repetitions, while a big error (e.g. 0.5) leads to the 
opposite behaviour.
- output_width and output_height: width and height of the output image.
- destination: path to the folder where to store the output. If not 
specified, the folder of the input file is taken. Output file name: 
<input_file_name>_result.png

Additional basic parameters:

- input_scale: scale to apply to the input texture before launching the
quilting computation. This parameter is useful when the input image is 
big, and it would take a long time to compute, or when the desired level
of details of the output image is smaller (or larger) than the one of 
input image. (default: 1)
- constraint_start: constraint the first tile (up left corner) to be the
same of the input image


Customization
-------------
- **multiple images**: it is possible to provide a list of related
images (color, bump, spec, etc) as input. The computation of the first 
one will be extended to the following: the output will be a set of 
images with the same mutual relations (color, bump. spec, etc.).
 *How*: provide a generic path. E.g.: 
    ```
    quilt C:/data/tiles/tiles*.jpg
    ```
    this will indicate to consider all the images in the folder whose 
    name respect the given pattern (e.g. tiles.jpg, tiles_bump.jpg)
    
  ![Alt text](data/figures/layers.png?raw=true "Layers")

- **remove areas**: it is possible to provide a mask (black and white 
image) of the input texture where black regions indicate areas that 
should not appear in the result, while areas to be considered are left
white. It can be used to remove areas that would appear repeating in the
output image (e.g. a brick of a different color in a brick wall).
Before starting the computation, the given mask is edited in the 
following ways: 
    - "white" values are replaced with 0, while "black" (masked) values
    are set to infinite. In this way, during the patch matching step, 
    the mask is summed to the distance matrix before searching for its 
    minima. Doing so, masked values (infinite values) are not chosen. 
    - every masked area is expanded in order to prevent the algorithm to 
    choose all tiles that contain at least one masked pixel.
 *How*: add the option --input_mask followed by the masks's path. E.g.:
    ```
    quilt C:/data/tiles/tiles*.jpg --input_mask C:/data/mask.jpg
    ``` 

- **output boundaries**: it is possible to provide a mask (black and
white image) of the output image where white regions indicate areas to 
cover with texture, while back areas are desired to be filled with
background.
 *How*: add the option --cut_mask followed by the masks's path.
    ```
    quilt C:/data/tiles/tiles*.png --cut_mask C:/data/mask.jpg
    ``` 
    
    Notes: 
    
    * in this case it is necessary that the input image contains some 
    background regions. In order to have better results, the background
    has to be just one flat color, possibly distinct form the foreground
    color (e.g. black or white). 
    * the boundaries are an indication, they will not be precisely 
     reproduced.


Getting more variations:
------------------------
In order to get more variations in the output image it is possible to 
source the tiles not only from the input image, but also from its 
rotations or flipped versions. 
Command line parameters:

- **rotations**: number of rotations of 90 degrees to apply to the input 
image before the computation. Possible values are: 0, 2 (0' and 180') 
and 4 (0', 90', 180', 270'). (default: 0)
If this option is selected, a new source image is created, combining 
together the input image and its rotation/s. The images are organized as
shown in the drawing (the extra area in the four rotations is left at 0):

![Alt text](data/figures/rotations.jpg?raw=true "Rotations Schema")


- **flip**: boolean tuple for (flip_vertical, flip_horizontal). The
image if flipped in the following way:

![Alt text](data/figures/flips.jpg?raw=true "Rotations mask")
       
When rotation or flip is performed, an input mask is created which masks
the empty areas and the edges between the images in the source. In this 
way, no patch containing parts of different images is be sourced during
the synthesis process. Here is an example of mask derived from a four-
rotation. Notice that is the user provides a custom input mask, the two
are added together. 

![Alt text](data/figures/rotations_mask.jpg?raw=true "Rotations mask")
           
Usage notes:

- this goodness of these features is strictly related to the nature of 
the input texture: usually, if the texture is oriented (anisotropic) the
use of these features is not advised.
- these feature slow down the computation time.


Performance
-----------
It is possible perform a single process computation or a multi-process 
computation. 

**Overview of the multi-process computation**
The first process computes quilting with big tiles. Once done, it 
creates new processes (one per available core) and assigns a big-tile to 
each of them. Each sub-process then performs the quilting computation 
with regular-sized tiles. Once all the big tiles have been processed, 
the results are seamlessly patched. 
Command line parameters:

- multiprocess: performs the multiprocessing computation (default: True)
- cores: number of cores available for quilting (default: number of 
cores in the machine minus two)
- big_tilesize: size of the big tiles. Note that if the number of big 
tiles that fit in the destination image (considering overlapping) is the 
same of cores, a better performance will be achieved. (default: 500)
- big_overlap: size of the overlap for the big tiles. (default: 100)


Logging
-------
Logging is currenlt on StandardError only. It is customized in order to use the 
styles of Click.echo. Moreover, it is multi-threading in order to be used by
multi-processes in Quilt.


Licence
=======
This software is licensed under the GPL.









    


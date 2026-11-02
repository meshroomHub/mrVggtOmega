__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class VGGTOmegaCompute(desc.Node):
    category = "vggt"

    documentation = """ A wrapper for VGGT-Omega method to extract poses and depthmaps """

    gpu = desc.Level.INTENSIVE

    size = desc.DynamicNodeSize('input')

    inputs = [
        desc.File(
            name="input",
            label="Input SfmData",
            description="Input images to process. Folder path or sfmData filepath",
            value="",
        ),
        desc.IntParam(
            name="imageResolution",
            label="Image Resolution",
            description="Network input resolution (must be a multiple of 16).",
            value=512,
            range=(64, 1024, 16),
        ),
        desc.ChoiceParam(
            name="resizeMode",
            label="Resize Mode",
            description="balanced: keep the token count close to resolution^2.\n"
                        "max_size: resize the longest side to resolution.",
            values=["balanced", "max_size"],
            value="balanced",
        ),
        desc.BoolParam(
            name="sharedIntrinsic",
            label="Shared Intrinsic",
            description="Keep one intrinsic per input intrinsic group, using the median of the per-view estimated focals.\n"
                        "If disabled, each view gets its own intrinsic.",
            value=False,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output SfmData',
            description="Output SfmData.",
            value="{nodeCacheFolder}/output.usda",
        ),
        desc.File(
            name='outputDirectory',
            label='Output Directory',
            description="Output",
            value="{nodeCacheFolder}",
        )
    ]

    def processChunk(self, chunk):
        from pyalicevision import sfmData
        from pyalicevision import sfmDataIO
        from pyalicevision import geometry as geo
        from pyalicevision import camera as cam
        from pyalicevision import image as avimg

        import logging
        import os
        import torch
        import numpy as np
        import pathlib
        import sys
        from PIL import Image

        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera
        from vggt_omega.utils.load_fn import _balanced_target_shape, _max_size_target_shape

        from torchvision import transforms as TF

        patch_size = 16

        def load_image(image_path, image_resolution, mode):

            ret = {}

            # Open image
            optRead = avimg.ImageReadOptions(avimg.EImageColorSpace_NO_CONVERSION)
            image = avimg.Image_RGBColor()
            avimg.readImage(image_path, image, optRead)
            img = Image.fromarray(image.getNumpyArray())

            metadata = avimg.readImageMetadataAsMap(image_path)
            pixelAspectRatio = float(metadata['PixelAspectRatio']) if 'PixelAspectRatio' in metadata else 1.0
            orientation = int(metadata['Orientation']) if 'Orientation' in metadata else 1
            compression = metadata['compression'] if 'compression' in metadata else "zips"

            # Crop horizontally to aspect ratio [0.5, 2.0] (centered) if needed
            width, height = img.size
            aspect_ratio = height / max(width, 1)
            min_aspect, max_aspect = 0.5, 2.0
            if aspect_ratio < min_aspect:
                crop_width = int(round(height / min_aspect))
                left = (width - crop_width) // 2
                img = img.crop((left, 0, left + crop_width, height))
                width = crop_width
            elif aspect_ratio > max_aspect:
                crop_width = int(round(height / max_aspect))
                left = (width - crop_width) // 2
                img = img.crop((left, 0, left + crop_width, height))
                width = crop_width

            # Aspect preserving resize to a multiple of patch size
            aspect_ratio = height / max(width, 1)
            if mode == "balanced":
                new_height, new_width = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
            else:
                new_height, new_width = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)

            # Resize with new dimensions (width, height)
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

            ret["path"] = image_path
            ret["image"] = img
            ret["pixelAspectRatio"] = pixelAspectRatio
            ret["orientation"] = orientation
            ret["compression"] = compression
            ret["crop_left"] = left if aspect_ratio < 0.5 or aspect_ratio > 2.0 else 0
            ret["crop_width"] = crop_width if aspect_ratio < 0.5 or aspect_ratio > 2.0 else width

            return ret

        def computeImagesPadding(imagesInfos):

            sizes = set()

            #Upgrade image to tensor
            for idview, imageInfo in imagesInfos.items():
                img = imageInfo["image"]
                sizes.add(img.size)

            # Find maximum dimensions

            max_width = max(size[0] for size in sizes)
            max_height = max(size[1] for size in sizes)

            #Compute padding for all images
            for idview, imageInfo in imagesInfos.items():
                img = imageInfo["image"]

                w_padding = max_width - img.size[0]
                h_padding = max_height - img.size[1]

                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                imageInfo["padding"] = (pad_top, pad_bottom, pad_left, pad_right)

            # Apply padding
            to_tensor = TF.ToTensor()
            for idview, imageInfo in imagesInfos.items():
                img = imageInfo["image"]
                img = to_tensor(img)
                top, bottom, left, right = imageInfo["padding"]
                img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=1.0)
                imageInfo["image"] = img

            return imagesInfos


        def generatePose(input):
            R = input[:3,:3]
            t = input[:3,3]
            c = -R.transpose() @ t
            R = R.astype('d')
            c = c.astype('d')
            transform = geo.Pose3(R, c)
            return sfmData.CameraPose(transform)

        def main(chunk):
            levels = {
                "fatal": logging.CRITICAL,
                "error": logging.ERROR,
                "warning": logging.WARNING,
                "info": logging.INFO,
                "debug": logging.DEBUG,
                "trace": logging.DEBUG,
            }
            logger = logging.getLogger("VGGTOmegaCompute")
            logger.setLevel(levels.get(chunk.node.verboseLevel.value, logging.INFO))
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
                logger.addHandler(handler)
            logger.propagate = False

            # prepare output
            outputDirectory = pathlib.Path(chunk.node.outputDirectory.value)
            os.makedirs(name=outputDirectory, exist_ok=True)

            imageResolution = chunk.node.imageResolution.value
            if imageResolution % patch_size != 0:
                logger.error(f"Image resolution must be a multiple of {patch_size}")
                sys.exit(1)

            # Initialize cuda
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available():
                logger.error("CUDA is not available. Check your environment.")
                sys.exit(1)

            # Parsing SfmData
            dataAV = sfmData.SfMData()
            if not sfmDataIO.load(dataAV, chunk.node.input.value, sfmDataIO.ALL):
                logger.error("Impossible to load sfmData")
                sys.exit(1)

            views = dataAV.getViews()
            poses = dataAV.getPoses()
            intrinsics = dataAV.getIntrinsics()

            # Find all images in sfmData
            imagesInfos = dict()
            for id, v in views.items():
                imagesInfos[id] = load_image(v.getImage().getImagePath(), imageResolution, chunk.node.resizeMode.value)

            # Handle multiple images sizes
            imagesInfos = computeImagesPadding(imagesInfos)

            # Transform to unique tensor
            images = []
            for id, imageInfo in imagesInfos.items():
                images.append(imageInfo["image"])
            images = torch.stack(images)

            images = images.to(device)

            # Loading model
            logger.info("Loading model.")
            model = VGGTOmega()
            model.load_state_dict(torch.load(os.getenv("VGGT_OMEGA_MODEL_PATH"), map_location="cpu"))
            model = model.to(device).eval()

            # Run inference (autocast is handled by the model)
            logger.info("Running inference.")
            with torch.inference_mode():
                predictions = model(images)

            logger.info("Inference finished.")

            # Convert pose encoding to extrinsic and intrinsic matrices
            extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            predictions.pop("images", None)

            # Convert tensors to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

            # Copy original intrinsics to be scattered
            bckIntrinsics = {}
            for id, intrinsic in intrinsics.items():
                if not cam.isPinhole(intrinsic.getType()):
                    logger.error("Invalid camera type")
                    sys.exit(1)
                bckIntrinsics[id] = intrinsic

            twidth = images.shape[-1]
            theight = images.shape[-2]

            # Estimated focals per view, grouped by original intrinsic
            focals = {}

            for pos, (id, v) in enumerate(views.items()):

                iinfo = imagesInfos[id]

                # Retrieve padding
                top, bottom, left, right = iinfo["padding"]

                # Copy extrinsic to sfmData
                poses[id] = generatePose(predictions["extrinsic"][pos])

                # Duplicate intrinsic as VGGT-Omega defines a focal per view
                K = predictions["intrinsic"][pos]

                # ATM, K is a fake calibration matrix where only the scale
                # coefficient is truly estimated.
                # We are just upscaling the scale part to reflect the image resize
                # (padding does not change the focal in pixels)
                orig_width = v.getImage().getWidth()
                orig_height = v.getImage().getHeight()

                # Account for horizontal crop (centered)
                crop_left = iinfo["crop_left"]
                crop_width = iinfo["crop_width"]

                # Scale: original_image -> cropped -> resized_padded -> unpadded
                scalex = crop_width / (twidth - left - right)
                scaley = orig_height / (theight - top - bottom)
                fx = K[0, 0] * scalex
                fy = K[1, 1] * scaley

                focals.setdefault(v.getIntrinsicId(), []).append((id, fx, fy))

                frameName = pathlib.Path(iinfo["path"]).stem

                top_idx = top
                bottom_idx = theight - bottom if bottom > 0 else theight
                left_idx = left
                right_idx = twidth - right if right > 0 else twidth

                # Retrieve depth and unpad it (should not change intrinsics)
                depth = np.ascontiguousarray(predictions["depth"][pos, top_idx:bottom_idx, left_idx:right_idx, :])
                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                image = avimg.Image_float()
                image.fromNumpyArray(depth)

                outname = str(outputDirectory / ("depth_" + frameName))
                oiio_params = avimg.oiioParams(iinfo["orientation"], iinfo["pixelAspectRatio"], iinfo["compression"])
                avimg.writeImage(outname + ".exr", image, optWrite, oiio_params.get())

            # Create the new pinhole intrinsics
            intrinsics.clear()
            sharedIntrinsic = chunk.node.sharedIntrinsic.value
            for intrinsicId, groupFocals in focals.items():
                if sharedIntrinsic:
                    # One intrinsic per original group, median is robust to outlier views
                    fx = float(np.median([f[1] for f in groupFocals]))
                    fy = float(np.median([f[2] for f in groupFocals]))
                    groupFocals = [(viewId, fx, fy) for viewId, _, _ in groupFocals]

                for viewId, fx, fy in groupFocals:
                    newId = intrinsicId if sharedIntrinsic else viewId
                    if newId not in intrinsics:
                        intrinsic = cam.Pinhole.cast(bckIntrinsics[intrinsicId].clone())
                        intrinsic.setScale(np.array([fx, fy], dtype='d'))
                        intrinsics[newId] = intrinsic
                    views[viewId].setIntrinsicId(newId)


            torch.cuda.empty_cache()
            sfmDataIO.save(dataAV, chunk.node.output.value, sfmDataIO.ALL)

        # Process chunk entry point
        main(chunk)

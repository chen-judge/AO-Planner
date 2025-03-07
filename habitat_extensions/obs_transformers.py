import copy
import numbers
from typing import Dict, List, Tuple, Union

import torch
from gym import spaces
from habitat.config import Config
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import *
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    overwrite_gym_box_shape,
)
from torch import Tensor


@baseline_registry.register_obs_transformer()
class CenterCropperPerSensor(ObservationTransformer):
    """An observation transformer that center crops your input on a per-sensor basis."""

    sensor_crops: Dict[str, Union[int, Tuple[int, int]]]
    channels_last: bool

    def __init__(
        self,
        sensor_crops: List[Tuple[str, Union[int, Tuple[int, int]]]],
        channels_last: bool = True,
    ):
        """Args:
        size: A sequence (h, w) or int of the size you wish to resize/center_crop.
                If int, assumes square crop
        channels_list: indicates if channels is the last dimension
        trans_keys: The list of sensors it will try to centercrop.
        """
        super().__init__()

        self.sensor_crops = dict(sensor_crops)
        for k in self.sensor_crops:
            size = self.sensor_crops[k]
            if isinstance(size, numbers.Number):
                self.sensor_crops[k] = (int(size), int(size))
            assert len(size) == 2, "forced input size must be len of 2 (h, w)"

        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        observation_space = copy.deepcopy(observation_space)
        for key in observation_space.spaces:
            if (
                key in self.sensor_crops
                and observation_space.spaces[key].shape[-3:-1]
                != self.sensor_crops[key]
            ):
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                logger.info(
                    # "Center cropping observation size of %s from %s to %s"
                    "Center cropping observation size of %s from %s to %s\nHowever, based on the saved results, it seems that there has been no impact on the observation..."
                    % (key, (h, w), self.sensor_crops[key])
                )

                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.sensor_crops[key]
                )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        observations.update(
            {
                sensor: center_crop(
                    observations[sensor],
                    self.sensor_crops[sensor],
                    channels_last=self.channels_last,
                )
                for sensor in self.sensor_crops
                if sensor in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR
        return cls(cc_config.SENSOR_CROPS)

@baseline_registry.register_obs_transformer()
class ResizerPerSensor(ObservationTransformer):
    r"""An nn module the resizes images to any aspect ratio.
    This module assumes that all images in the batch are of the same size.
    """

    def __init__(
        self,
        sizes: int,
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        super().__init__()
        """Args:
        size: The size you want to resize
        channels_last: indicates if channels is the last dimension
        """
        self.sensor_resizes = dict(sizes)
        for k in self.sensor_resizes:
            size = self.sensor_resizes[k]
            if isinstance(size, numbers.Number):
                self.sensor_resizes[k] = (int(size), int(size))
            assert len(size) == 2, "forced input size must be len of 2 (h, w)"

        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):

        for key in observation_space.spaces:
            if (
                key in self.sensor_resizes
                and observation_space.spaces[key].shape[-3:-1]
                != self.sensor_resizes[key]
            ):
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                logger.info(
                    "Resizing observation size of %s from %s to %s"
                    % (key, (h, w), self.sensor_resizes[key])
                )

                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.sensor_resizes[key]
                )

        return observation_space

    def _transform_obs(self, obs: torch.Tensor, size) -> torch.Tensor:
        img = torch.as_tensor(obs)
        no_batch_dim = len(img.shape) == 3
        if len(img.shape) < 3 or len(img.shape) > 5:
            raise NotImplementedError()
        if no_batch_dim:
            img = img.unsqueeze(0)  # Adds a batch dimension
        h, w = get_image_height_width(img, channels_last=self.channels_last)
        if self.channels_last:
            if len(img.shape) == 4:
                # NHWC -> NCHW
                img = img.permute(0, 3, 1, 2)
            else:
                # NDHWC -> NDCHW
                img = img.permute(0, 1, 4, 2, 3)

        h, w = size
        img = torch.nn.functional.interpolate(
            img.float(), size=(h, w), mode="area"
        ).to(dtype=img.dtype)
        if self.channels_last:
            if len(img.shape) == 4:
                # NCHW -> NHWC
                img = img.permute(0, 2, 3, 1)
            else:
                # NDCHW -> NDHWC
                img = img.permute(0, 1, 3, 4, 2)
        if no_batch_dim:
            img = img.squeeze(dim=0)  # Removes the batch dimension
        return img

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        observations.update(
            {
                sensor: self._transform_obs(
                    observations[sensor], self.sensor_resizes[sensor])
                for sensor in self.sensor_resizes
                if sensor in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        r_config = config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR
        return cls(r_config.SIZES)


class Cube2Equirect(ProjectionConverter):
    """Just to compute depth equirect with height same as equ_h"""

    def __init__(self, equ_h: int, equ_w: int):
        """Args:
        equ_h: (int) the height of the generated equirect
        equ_w: (int) the width of the generated equirect
        """

        # Cubemap input
        input_projections = get_cubemap_projections(equ_h,equ_h)

        # Equirectangular output
        output_projection = EquirectProjection(equ_h, equ_w)
        super(Cube2Equirect, self).__init__(
            input_projections, output_projection
        )

@baseline_registry.register_obs_transformer()
class CubeMap2Equirect(ProjectionTransformer):
    r"""This is an experimental use of ObservationTransformer that converts a cubemap
    output to an equirectangular one through projection. This needs to be fed
    a list of 6 cameras at various orientations but will be able to stitch a
    360 sensor out of these inputs. The code below will generate a config that
    has the 6 sensors in the proper orientations. This code also assumes a 90
    FOV.

    Sensor order for cubemap stiching is Back, Down, Front, Left, Right, Up.
    The output will be writen the UUID of the first sensor.
    """

    def __init__(
        self,
        sensor_uuids: List[str],
        eq_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):
        r""":param sensor_uuids: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param eq_shape: The shape of the equirectangular output (height, width)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
        :param depth_key: If sensor_uuids has depth_key substring, they are processed as depth
        """

        converter = Cube2Equirect(eq_shape[0], eq_shape[1])
        super(CubeMap2Equirect, self).__init__(
            converter,
            sensor_uuids,
            eq_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        cube2eq_config = config.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ
        if hasattr(cube2eq_config, "TARGET_UUIDS"):
            # Optional Config Value to specify target UUID
            target_uuids = cube2eq_config.TARGET_UUIDS
        else:
            target_uuids = None
        return cls(
            cube2eq_config.SENSOR_UUIDS,
            eq_shape=(
                cube2eq_config.HEIGHT,
                cube2eq_config.WIDTH,
            ),
            target_uuids=target_uuids,
        )
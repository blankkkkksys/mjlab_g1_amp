"""Keyboard teleop for UniformVelocityCommand during play."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable

import mujoco
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import EnvProtocol, PolicyProtocol
from mjlab.viewer.native.keys import (
  KEY_A,
  KEY_D,
  KEY_DOWN,
  KEY_E,
  KEY_LEFT,
  KEY_Q,
  KEY_RIGHT,
  KEY_S,
  KEY_UP,
  KEY_W,
  KEY_X,
  KEY_Z,
)


@dataclass
class _GuiFlag:
  value: bool | float


def has_twist_command(env: EnvProtocol) -> bool:
  return "twist" in env.unwrapped.command_manager.active_terms


class KeyboardTwistController:
  """Map WASD / arrow keys to velocity command overrides."""

  def __init__(
    self,
    twist: UniformVelocityCommand,
    *,
    lin_step: float = 0.1,
    ang_step: float = 0.1,
  ) -> None:
    self._twist = twist
    self._lin_step = lin_step
    self._ang_step = ang_step
    self._lock = Lock()
    self._vx = 0.0
    self._vy = 0.0
    self._wz = 0.0
    self._get_env_idx: Callable[[], int] | None = None
    ranges = twist.cfg.ranges
    self._vx_range = ranges.lin_vel_x
    self._vy_range = ranges.lin_vel_y
    self._wz_range = ranges.ang_vel_z

  @classmethod
  def from_env(
    cls,
    env: EnvProtocol,
    *,
    command_name: str = "twist",
    lin_step: float = 0.1,
    ang_step: float = 0.1,
  ) -> KeyboardTwistController:
    twist = env.unwrapped.command_manager.get_term(command_name)
    if not isinstance(twist, UniformVelocityCommand):
      raise TypeError(f"Command {command_name!r} is not UniformVelocityCommand")
    return cls(twist, lin_step=lin_step, ang_step=ang_step)

  def install(self, get_env_idx: Callable[[], int]) -> None:
    """Hook keyboard values into UniformVelocityCommand joystick override."""
    self._get_env_idx = get_env_idx
    if self._twist._joystick_sliders:
      if self._twist._joystick_enabled is not None:
        self._twist._joystick_enabled.value = True
    else:
      self._twist._joystick_enabled = _GuiFlag(True)
      self._twist._joystick_sliders = [_GuiFlag(0.0), _GuiFlag(0.0), _GuiFlag(0.0)]
      self._twist._joystick_get_env_idx = get_env_idx
    self._sync_sliders()
    self.print_help()

  def print_help(self) -> None:
    print(
      "[INFO] Keyboard velocity control enabled:\n"
      "  W / Up    : forward (+vx)\n"
      "  S / Down  : backward (-vx)\n"
      "  A / Left  : turn left (+wz)\n"
      "  D / Right : turn right (-wz)\n"
      "  Q / E     : strafe left / right (if enabled in command ranges)\n"
      "  Z / X     : zero velocity"
    )

  def zero(self) -> None:
    with self._lock:
      self._vx = 0.0
      self._vy = 0.0
      self._wz = 0.0
    self._sync_sliders()

  def on_key(self, key: int) -> None:
    """Handle MuJoCo viewer key presses (runs on viewer thread)."""
    with self._lock:
      if key in (KEY_W, KEY_UP):
        self._vx = self._clamp(self._vx + self._lin_step, self._vx_range)
      elif key in (KEY_S, KEY_DOWN):
        self._vx = self._clamp(self._vx - self._lin_step, self._vx_range)
      elif key in (KEY_A, KEY_LEFT):
        self._wz = self._clamp(self._wz + self._ang_step, self._wz_range)
      elif key in (KEY_D, KEY_RIGHT):
        self._wz = self._clamp(self._wz - self._ang_step, self._wz_range)
      elif key == KEY_Q:
        self._vy = self._clamp(self._vy + self._lin_step, self._vy_range)
      elif key == KEY_E:
        self._vy = self._clamp(self._vy - self._lin_step, self._vy_range)
      elif key in (KEY_Z, KEY_X):
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
      else:
        return
    self._sync_sliders()

  def _sync_sliders(self) -> None:
    sliders = self._twist._joystick_sliders
    if not sliders:
      return
    with self._lock:
      vx, vy, wz = self._vx, self._vy, self._wz
    sliders[0].value = vx
    sliders[1].value = vy
    sliders[2].value = wz

  @staticmethod
  def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return float(max(bounds[0], min(bounds[1], value)))

  @property
  def command_text(self) -> str:
    with self._lock:
      return f"vx={self._vx:+.2f} vy={self._vy:+.2f} wz={self._wz:+.2f}"


class KeyboardTwistNativeViewer(NativeMujocoViewer):
  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    controller: KeyboardTwistController,
    **kwargs,
  ) -> None:
    self._keyboard = controller
    super().__init__(env, policy, key_callback=controller.on_key, **kwargs)
    self._keyboard.install(lambda: self.env_idx)

  def reset_environment(self) -> None:
    super().reset_environment()
    self._keyboard.zero()

  def _set_status_overlay(self, viewer) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    text_1 = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT\nCmd"
    text_2 = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)\n"
      f"{self._keyboard.command_text}"
    )
    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      text_1,
      text_2,
    )
    viewer.set_texts(overlay)


class KeyboardTwistViserViewer(ViserPlayViewer):
  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    controller: KeyboardTwistController,
    **kwargs,
  ) -> None:
    self._keyboard = controller
    super().__init__(env, policy, **kwargs)

  def setup(self) -> None:
    super().setup()
    self._keyboard.install(lambda: self._scene.env_idx)

  def reset_environment(self) -> None:
    super().reset_environment()
    self._keyboard.zero()

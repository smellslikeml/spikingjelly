from typing import Optional

import torch
import torch.nn as nn

from .. import surrogate
from .base_node import SimpleBaseNode

__all__ = ["TernaryIFNode", "TernaryLIFNode"]


class _TernaryBaseNode(SimpleBaseNode):
    # Shared ternary fire / reset / learnable spike amplitude for the public
    # TernaryIFNode and TernaryLIFNode; subclasses only implement the charge.

    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )
        # The layer-wise trainable spike amplitude ``a`` of Eq. (16) in
        # arXiv:2312.06372 (a scalar per layer). The paper does not prescribe
        # an initialization; 1.0 makes the trainable form start identical to
        # the fixed-amplitude ternary neuron of Eq. (10).
        self.scale = nn.Parameter(torch.tensor(1.0))

    def extra_repr(self):
        return (
            f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, "
            f"detach_reset={self.detach_reset}, step_mode={self.step_mode}"
        )

    def neuronal_fire(self) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <TernaryBaseNode.neuronal_fire-cn>` | :ref:`English <TernaryBaseNode.neuronal_fire-en>`

        ----

        .. _TernaryBaseNode.neuronal_fire-cn:

        * **中文**

        三值发放（论文式 (10) 与 (16)）：前向为带符号阶跃，输出 ``scale * b``，
        其中 ``b`` 取 ``{-1, 0, +1}``；反向通过 ``surrogate_function`` 在
        正负两个阈值处均回传替代梯度。

        :return: 当前时间步的（乘以可学习幅值后的）三值脉冲
        :rtype: torch.Tensor

        ----

        .. _TernaryBaseNode.neuronal_fire-en:

        * **English**

        Ternary fire (Eqs. (10) and (16) of the paper): the forward pass is a
        signed Heaviside emitting ``scale * b`` with ``b`` in ``{-1, 0, +1}``;
        the backward pass routes the surrogate gradient of
        ``surrogate_function`` through both the positive and negative
        thresholds.

        :return: Ternary spike of the current time step, multiplied by the
            learnable amplitude
        :rtype: torch.Tensor
        """
        fire_pos = self.surrogate_function(self.v - self.v_threshold)
        fire_neg = self.surrogate_function(-self.v - self.v_threshold)
        return self.scale * (fire_pos - fire_neg)

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        r"""
        **API Language** - :ref:`中文 <TernaryBaseNode.neuronal_reset-cn>` | :ref:`English <TernaryBaseNode.neuronal_reset-en>`

        ----

        .. _TernaryBaseNode.neuronal_reset-cn:

        * **中文**

        任意符号发放后重置（论文式 (9) 的 :math:`(1-|o|)` 因子）：无论发放
        ``+1`` 还是 ``-1``，发放位置都会被重置。重置使用与 ``scale`` 无关的
        归一化发放掩码 ``spike / scale``：``v_reset`` 不为 ``None`` 时硬重置到
        ``v_reset``，为 ``None`` 时执行带符号软重置
        :math:`V[t] = H[t] - V_{threshold} \cdot B[t]`。

        :param spike: 当前时间步的输出脉冲（已乘 ``self.scale``）
        :type spike: torch.Tensor

        ----

        .. _TernaryBaseNode.neuronal_reset-en:

        * **English**

        Reset after a spike of either sign (the :math:`(1-|o|)` factor of
        Eq. (9) in the paper): a firing site is reset whether it emitted ``+1``
        or ``-1``. The reset uses the scale-invariant normalized firing mask
        ``spike / scale``: sites are hard-reset to ``v_reset`` when it is not
        ``None``, and the signed soft reset
        :math:`V[t] = H[t] - V_{threshold} \cdot B[t]` is applied otherwise.

        :param spike: Output spikes of the current time step (already
            multiplied by ``self.scale``)
        :type spike: torch.Tensor
        """
        b = spike / self.scale
        if self.detach_reset:
            b = b.detach()

        if self.v_reset is None:
            # signed soft reset
            self.v = self.v - self.v_threshold * b
        else:
            # hard reset on either signed spike
            fired = b.abs()
            self.v = fired * self.v_reset + (1.0 - fired) * self.v


class TernaryIFNode(_TernaryBaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        r"""
        **API Language** - :ref:`中文 <TernaryIFNode.__init__-cn>` | :ref:`English <TernaryIFNode.__init__-en>`

        ----

        .. _TernaryIFNode.__init__-cn:

        * **中文**

        三值脉冲 Integrate-and-Fire 神经元，发放 ``{-1, 0, +1}`` 三值脉冲而非
        ``{0, 1}`` 二值脉冲，来自 *Ternary Spike: Learning Ternary Spikes for
        Spiking Neural Networks* (Guo et al., AAAI 2024,
        `arXiv:2312.06372 <https://arxiv.org/abs/2312.06372>`_)，对应 issue
        `#757 <https://github.com/fangwei123456/spikingjelly/issues/757>`_。

        本实现是依据论文方法部分（式 (9)-(10)、(15)-(16)、(18)-(19)）的
        **净室重构**：作者的参考仓库未附带开源许可证，未被阅读或使用。

        阈下充电动力学与 :class:`IFNode` 一致：

        .. math::

            H[t] = V[t-1] + X[t]

        三值发放（论文式 (10)，可学习形式为式 (16)）：

        .. math::

            B[t] = \begin{cases} 1, & H[t] \geq V_{threshold} \\ -1, & H[t] \leq -V_{threshold} \\ 0, & \text{其它} \end{cases}

        .. math::

            O[t] = a \cdot B[t]

        其中逐层可学习幅值 ``a`` 即 :attr:`scale`（论文中
        :math:`a \in \mathbb{R}^{1 \times 1 \times 1}`，即每层一个标量）。前向
        通过双侧替代函数之差 :math:`\Theta(H[t] - V_{threshold}) -
        \Theta(-H[t] - V_{threshold})` 计算带符号阶跃，反向在正负两个阈值处均
        使用 ``surrogate_function`` 的替代梯度。推理时可按论文式 (18)-(19) 的
        重参数化技巧，将 ``a`` 折叠进消费本层脉冲的权重（如
        ``fc.weight *= scale``，即 :math:`\tilde{K} = a \cdot K`）并把
        ``scale`` 置回 1，即可在不改变网络输出的前提下恢复纯 ``{-1, 0, +1}``
        的事件驱动三值脉冲。

        重置遵循“任意符号发放后重置”（论文式 (9) 的 :math:`(1-|o|)` 因子）：
        发放 ``+1`` 或 ``-1`` 的位置都会被重置——``v_reset`` 不为 ``None`` 时
        硬重置到 ``v_reset``，为 ``None`` 时执行带符号软重置
        :math:`V[t] = H[t] - V_{threshold} \cdot B[t]`。

        本类基于 :class:`SimpleBaseNode` 的纯 PyTorch 充电-放电-重置接口实现，
        仅支持 torch 后端；cupy/triton 加速留作后续工作。参数化 ``tau`` 的
        三值变体（PLIF 等）同样超出本次范围。

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float
        :param v_reset: 神经元的重置电压。如果不为 ``None``，发放（任意符号）后
            电压会被重置为 ``v_reset``；如果设置为 ``None``，发放后电压会减去
            ``v_threshold`` 乘以脉冲符号
        :type v_reset: Optional[float]
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数，
            正负两个阈值处均会使用
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否将 reset 过程的计算图分离
        :type detach_reset: bool
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _TernaryIFNode.__init__-en:

        * **English**

        The ternary-spike Integrate-and-Fire neuron, which emits ternary
        ``{-1, 0, +1}`` spikes instead of binary ``{0, 1}`` ones, from
        *Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks*
        (Guo et al., AAAI 2024,
        `arXiv:2312.06372 <https://arxiv.org/abs/2312.06372>`_), requested in
        issue
        `#757 <https://github.com/fangwei123456/spikingjelly/issues/757>`_.

        This is a **clean-room reconstruction** from the method section of the
        paper (Eqs. (9)-(10), (15)-(16), (18)-(19)); the authors' reference
        repository carries no open-source license and was not read or used.

        The sub-threshold charge is identical to :class:`IFNode`:

        .. math::

            H[t] = V[t-1] + X[t]

        Ternary fire (Eq. (10); trainable form in Eq. (16)):

        .. math::

            B[t] = \begin{cases} 1, & H[t] \geq V_{threshold} \\ -1, & H[t] \leq -V_{threshold} \\ 0, & \text{otherwise} \end{cases}

        .. math::

            O[t] = a \cdot B[t]

        where the layer-wise learnable amplitude ``a`` is :attr:`scale` (a
        scalar per layer, :math:`a \in \mathbb{R}^{1 \times 1 \times 1}` in the
        paper). The forward pass computes the signed Heaviside as the
        two-sided surrogate difference :math:`\Theta(H[t] - V_{threshold}) -
        \Theta(-H[t] - V_{threshold})`, and the backward pass applies the
        surrogate gradient of ``surrogate_function`` at both thresholds. At
        inference, the re-parameterization of Eqs. (18)-(19) folds ``a`` into
        the weights that consume this layer's spikes (e.g.
        ``fc.weight *= scale``, i.e. :math:`\tilde{K} = a \cdot K`) and resets
        ``scale`` to 1, restoring pure event-driven ``{-1, 0, +1}`` ternary
        spikes without changing the network outputs.

        The reset follows "reset after a spike of either sign" (the
        :math:`(1-|o|)` factor of Eq. (9)): a site firing ``+1`` or ``-1`` is
        reset — hard-reset to ``v_reset`` when it is not ``None``, otherwise
        the signed soft reset :math:`V[t] = H[t] - V_{threshold} \cdot B[t]`.

        The class is built on the pure-PyTorch charge-fire-reset interface of
        :class:`SimpleBaseNode` and supports the torch backend only; cupy and
        triton acceleration are left as future work. Parametrized-``tau``
        (e.g. PLIF) ternary variants are likewise out of scope.

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float
        :param v_reset: reset voltage of this neurons layer. If not ``None``,
            the voltage of a firing site (either sign) is set to ``v_reset``
            after a spike; if ``None``, ``v_threshold`` times the spike sign
            is subtracted after a spike
        :type v_reset: Optional[float]
        :param surrogate_function: the function for calculating surrogate
            gradients of the heaviside step function in backward, used at both
            the positive and negative thresholds
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: whether detach the computation graph of reset in
            backward
        :type detach_reset: bool
        :param step_mode: the step mode, which can be `s` (single-step) or `m`
            (multi-step)
        :type step_mode: str
        """
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )

    def neuronal_charge(self, x: torch.Tensor) -> None:
        r"""
        **API Language** - :ref:`中文 <TernaryIFNode.neuronal_charge-cn>` | :ref:`English <TernaryIFNode.neuronal_charge-en>`

        ----

        .. _TernaryIFNode.neuronal_charge-cn:

        * **中文**

        神经元充电的微分方程：

        .. math::

            H[t] = V[t-1] + X[t]

        :param x: 输入电压
        :type x: torch.Tensor
        :return: None（膜电位更新存储在 ``self.v`` 中）
        :rtype: None

        ----

        .. _TernaryIFNode.neuronal_charge-en:

        * **English**

        The differential equation for neuronal charge:

        .. math::

            H[t] = V[t-1] + X[t]

        :param x: Input voltage
        :type x: torch.Tensor
        :return: None (membrane potential is stored in ``self.v``)
        :rtype: None
        """
        self.v = self.v + x


class TernaryLIFNode(_TernaryBaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        r"""
        **API Language** - :ref:`中文 <TernaryLIFNode.__init__-cn>` | :ref:`English <TernaryLIFNode.__init__-en>`

        ----

        .. _TernaryLIFNode.__init__-cn:

        * **中文**

        三值脉冲 Leaky Integrate-and-Fire 神经元，发放 ``{-1, 0, +1}`` 三值脉冲
        而非 ``{0, 1}`` 二值脉冲，来自 *Ternary Spike: Learning Ternary Spikes
        for Spiking Neural Networks* (Guo et al., AAAI 2024,
        `arXiv:2312.06372 <https://arxiv.org/abs/2312.06372>`_)，对应 issue
        `#757 <https://github.com/fangwei123456/spikingjelly/issues/757>`_。
        论文本身采用的三值神经元即 LIF 形式（论文式 (9)/(15)，漏电因子
        :math:`\tau = 0.25`，对应本实现的 ``tau=4/3, decay_input=False,
        v_reset=0.0``）。

        本实现是依据论文方法部分（式 (9)-(10)、(15)-(16)、(18)-(19)）的
        **净室重构**：作者的参考仓库未附带开源许可证，未被阅读或使用。

        阈下充电动力学与 :class:`LIFNode` 一致。若 ``decay_input == True``:

        .. math::

            H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

        .. math::

            H[t] = V[t-1] - \frac{1}{\tau}(V[t-1] - V_{reset}) + X[t]

        三值发放、可学习幅值 :attr:`scale`（论文式 (16) 的逐层标量 ``a``）、
        推理期权重折叠（论文式 (18)-(19)）与“任意符号发放后重置”的重置规则
        均与 :class:`TernaryIFNode` 一致，详见其文档。

        本类基于 :class:`SimpleBaseNode` 的纯 PyTorch 充电-放电-重置接口实现，
        仅支持 torch 后端；cupy/triton 加速与参数化 ``tau`` 的三值变体（PLIF
        等）留作后续工作。

        :param tau: 膜电位时间常数
        :type tau: float
        :param decay_input: 输入是否也会参与衰减
        :type decay_input: bool
        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float
        :param v_reset: 神经元的重置电压。如果不为 ``None``，发放（任意符号）后
            电压会被重置为 ``v_reset``；如果设置为 ``None``，发放后电压会减去
            ``v_threshold`` 乘以脉冲符号。充电衰减以 0 为参考电位
        :type v_reset: Optional[float]
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数，
            正负两个阈值处均会使用
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否将 reset 过程的计算图分离
        :type detach_reset: bool
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _TernaryLIFNode.__init__-en:

        * **English**

        The ternary-spike Leaky Integrate-and-Fire neuron, which emits ternary
        ``{-1, 0, +1}`` spikes instead of binary ``{0, 1}`` ones, from
        *Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks*
        (Guo et al., AAAI 2024,
        `arXiv:2312.06372 <https://arxiv.org/abs/2312.06372>`_), requested in
        issue
        `#757 <https://github.com/fangwei123456/spikingjelly/issues/757>`_.
        The neuron used in the paper itself is the LIF form (Eqs. (9)/(15)
        with leak factor :math:`\tau = 0.25`, which corresponds to
        ``tau=4/3, decay_input=False, v_reset=0.0`` in this implementation).

        This is a **clean-room reconstruction** from the method section of the
        paper (Eqs. (9)-(10), (15)-(16), (18)-(19)); the authors' reference
        repository carries no open-source license and was not read or used.

        The sub-threshold charge is identical to :class:`LIFNode`. If
        ``decay_input == True``:

        .. math::

            H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))

        If ``decay_input == False``:

        .. math::

            H[t] = V[t-1] - \frac{1}{\tau}(V[t-1] - V_{reset}) + X[t]

        The ternary fire, the learnable amplitude :attr:`scale` (the layer-wise
        scalar ``a`` of Eq. (16)), the inference-time weight folding (Eqs.
        (18)-(19)) and the "reset after a spike of either sign" rule all match
        :class:`TernaryIFNode`; see its documentation for the equations.

        The class is built on the pure-PyTorch charge-fire-reset interface of
        :class:`SimpleBaseNode` and supports the torch backend only; cupy and
        triton acceleration and parametrized-``tau`` (e.g. PLIF) ternary
        variants are left as future work.

        :param tau: membrane time constant
        :type tau: float
        :param decay_input: whether the input will decay
        :type decay_input: bool
        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float
        :param v_reset: reset voltage of this neurons layer. If not ``None``,
            the voltage of a firing site (either sign) is set to ``v_reset``
            after a spike; if ``None``, ``v_threshold`` times the spike sign is
            subtracted after a spike. The charge decays toward 0
        :type v_reset: Optional[float]
        :param surrogate_function: the function for calculating surrogate
            gradients of the heaviside step function in backward, used at both
            the positive and negative thresholds
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: whether detach the computation graph of reset in
            backward
        :type detach_reset: bool
        :param step_mode: the step mode, which can be `s` (single-step) or `m`
            (multi-step)
        :type step_mode: str
        """
        assert isinstance(tau, float) and tau > 1.0
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )
        self.tau = tau
        self.decay_input = decay_input

    def extra_repr(self):
        return super().extra_repr() + f", tau={self.tau}"

    def neuronal_charge(self, x: torch.Tensor) -> None:
        r"""
        **API Language** - :ref:`中文 <TernaryLIFNode.neuronal_charge-cn>` | :ref:`English <TernaryLIFNode.neuronal_charge-en>`

        ----

        .. _TernaryLIFNode.neuronal_charge-cn:

        * **中文**

        神经元充电的微分方程。若 ``decay_input == True``:

        .. math::

            H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

        .. math::

            H[t] = V[t-1] - \frac{1}{\tau}(V[t-1] - V_{reset}) + X[t]

        :param x: 输入电压
        :type x: torch.Tensor
        :return: None（膜电位更新存储在 ``self.v`` 中）
        :rtype: None

        ----

        .. _TernaryLIFNode.neuronal_charge-en:

        * **English**

        The differential equation for neuronal charge. If
        ``decay_input == True``:

        .. math::

            H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))

        If ``decay_input == False``:

        .. math::

            H[t] = V[t-1] - \frac{1}{\tau}(V[t-1] - V_{reset}) + X[t]

        :param x: Input voltage
        :type x: torch.Tensor
        :return: None (membrane potential is stored in ``self.v``)
        :rtype: None
        """
        v_reset_value = 0.0 if self.v_reset is None else self.v_reset
        if self.decay_input:
            self.v = self.v + (x - (self.v - v_reset_value)) / self.tau
        else:
            self.v = self.v - (self.v - v_reset_value) / self.tau + x

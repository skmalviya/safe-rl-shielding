class Shield:

  def __init__(self):
    self.s2 = 0
    self.s1 = 0
    self.s0 = 0

  def tick(self, inputs):
    b = inputs[0]
    o3_1 = inputs[1]
    o2_1 = inputs[2]
    o1_1 = inputs[3]
    o3_2 = inputs[4]
    o2_2 = inputs[5]
    o1_2 = inputs[6]
    o3_3 = inputs[7]
    o2_3 = inputs[8]
    o1_3 = inputs[9]
    s2 = self.s2
    s1 = self.s1
    s0 = self.s0

    tmp11 = (1 if o1_3 else 0)
    tmp10 = (1 if o2_3 else tmp11)
    tmp9 = (tmp10 if o3_3 else 0)
    tmp8 = (1 if o1_2 else tmp9)
    tmp7 = (1 if o2_2 else tmp8)
    tmp6 = (tmp7 if o3_2 else 0)
    tmp5 = (1 if o1_1 else tmp6)
    tmp4 = (1 if o2_1 else tmp5)
    tmp3 = (tmp4 if o3_1 else 0)
    tmp12 = (1 if o3_1 else 0)
    tmp2 = (tmp3 if b else tmp12)
    tmp1 = (tmp2 if s1 else tmp12)
    o3__s = tmp1
    tmp21 = (1 - (1 if o2_3 else 0))
    tmp20 = (1 - (1 if o1_2 else tmp21))
    tmp19 = (1 if o2_2 else tmp20)
    tmp22 = (1 if o2_2 else 0)
    tmp18 = (1 - (tmp19 if o3_2 else tmp22))
    tmp17 = (1 - (1 if o1_1 else tmp18))
    tmp16 = (1 if o2_1 else tmp17)
    tmp23 = (1 if o2_1 else 0)
    tmp15 = (tmp16 if o3_1 else tmp23)
    tmp14 = (tmp15 if b else tmp23)
    tmp13 = (tmp14 if s1 else tmp23)
    o2__s = tmp13
    tmp28 = (1 if o1_1 else 0)
    tmp32 = (1 if o1_2 else 0)
    tmp34 = (1 if o1_3 else 0)
    tmp33 = (1 if o1_2 else tmp34)
    tmp31 = (tmp32 if o2_2 else tmp33)
    tmp30 = (tmp31 if o3_2 else tmp32)
    tmp29 = (1 if o1_1 else tmp30)
    tmp27 = (tmp28 if o2_1 else tmp29)
    tmp26 = (tmp27 if o3_1 else tmp28)
    tmp25 = (tmp26 if b else tmp28)
    tmp24 = (tmp25 if s1 else tmp28)
    o1__s = tmp24
    recovery__s = 0
    tmp46 = (1 if o1_3 else 0)
    tmp45 = (1 if o2_3 else tmp46)
    tmp44 = (tmp45 if o3_3 else 1)
    tmp43 = (1 if o1_2 else tmp44)
    tmp42 = (1 if o2_2 else tmp43)
    tmp41 = (tmp42 if o3_2 else 1)
    tmp40 = (1 if o1_1 else tmp41)
    tmp39 = (1 if o2_1 else tmp40)
    tmp38 = (tmp39 if o3_1 else 1)
    tmp37 = (tmp38 if b else 1)
    tmp36 = (tmp37 if s1 else 1)
    tmp35 = (1 - (1 if s2 else tmp36))
    s2n = tmp35
    tmp52 = (1 if o1_1 else 0)
    tmp51 = (1 if o2_1 else tmp52)
    tmp50 = (tmp51 if o3_1 else 1)
    tmp49 = (tmp50 if b else 1)
    tmp48 = (tmp49 if s0 else 1)
    tmp47 = (1 - (1 if s1 else tmp48))
    s1n = tmp47
    tmp58 = (1 if o1_1 else 0)
    tmp57 = (1 if o2_1 else tmp58)
    tmp56 = (tmp57 if o3_1 else 1)
    tmp55 = (tmp56 if b else 1)
    tmp54 = (1 if s0 else tmp55)
    tmp53 = (1 - (1 if s1 else tmp54))
    s0n = tmp53

    self.s2 = s2n
    self.s1 = s1n
    self.s0 = s0n

    return [ o3__s, o2__s, o1__s, recovery__s]

func.func @extf_e4m3(%x: f8E4M3FN) -> f32 {
    %0 = arith.extf %x : f8E4M3FN to f32
    return %0 : f32
}
func.func @extf_e8m0(%y: f8E8M0FNU) -> f32 {
    %0 = arith.extf %y : f8E8M0FNU to f32
    return %0 : f32
}
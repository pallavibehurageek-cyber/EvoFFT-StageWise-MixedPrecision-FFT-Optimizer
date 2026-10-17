import math

def float_to_fp8(val):
    """
    Custom FP8 (E4M3) encoding.
    Bias 7. 1 bit sign, 4 bits exponent, 3 bits mantissa.
    """
    if val == 0: return 0x00
    
    sign = 0x80 if val < 0 else 0x00
    val = abs(val)
    
    if val < 0.01: return sign 
    
    exponent_unbiased = math.floor(math.log2(val))
    exponent_stored = exponent_unbiased + 7
    
    if exponent_stored < 0: exponent_stored = 0
    if exponent_stored > 15: exponent_stored = 15
    
    mantissa_float = (val / (2**exponent_unbiased)) - 1.0
    mantissa_int = int(round(mantissa_float * 8)) 
    
    if mantissa_int == 8:
        mantissa_int = 0
        exponent_stored += 1
        
    return sign | ((exponent_stored & 0x0F) << 3) | (mantissa_int & 0x07)

def float_to_fp4(val):
    """
    Custom FP4 (E2M1) encoding.
    Bias = 1. Format: [Sign(1b), Mag(3b)]
    """
    sign = 0x8 if val < 0 else 0x0
    val = abs(val)
    
    if val < 0.25: mag = 0
    elif val < 0.75: mag = 1
    elif val < 1.25: mag = 2
    else: mag = 3
    
    return sign | mag

def generate_twiddles(filename="twiddles_24bit.txt", n_points=1024):
    """
    Generates unified 24-bit twiddle factors.
    Format: [FP8_R(8b)][FP8_I(8b)][FP4_R(4b)][FP4_I(4b)] = 24 bits total
    """
    with open(filename, "w") as f:
        # We only iterate n_points/2 times as per standard FFT twiddle requirements
        for k in range(n_points // 2):
            angle = -2 * math.pi * k / n_points
            real_part = math.cos(angle)
            imag_part = math.sin(angle)
            
            # Generate bit patterns
            fp8_r = float_to_fp8(real_part)
            fp8_i = float_to_fp8(imag_part)
            fp4_r = float_to_fp4(real_part)
            fp4_i = float_to_fp4(imag_part)
            
            # Construct the 24-bit string
            # [23:16] FP8 Real | [15:8] FP8 Imag | [7:4] FP4 Real | [3:0] FP4 Imag
            line = f"{fp8_r:08b}{fp8_i:08b}{fp4_r:04b}{fp4_i:04b}"
            
            f.write(line + "\n")
            
    print(f"File '{filename}' generated successfully.")
    print(f"Memory Layout: [FP8_Real][FP8_Imag][FP4_Real][FP4_Imag]")
    print(f"Total entries: {n_points // 2}")

if __name__ == "__main__":
    generate_twiddles()

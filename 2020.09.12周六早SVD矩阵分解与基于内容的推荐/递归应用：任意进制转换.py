def baseConvert(number,base):
    digits='0123456789ABCDEF'
    if number<base:
        return digits[number]
    else:
        return baseConvert(number//base,base)+digits[number%base]


print(baseConvert(232, 2))
print(baseConvert(495, 16))
print(baseConvert(459, 16))
print(baseConvert(10, 2))
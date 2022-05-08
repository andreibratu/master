function sum_p = ex1_4(s, e)
    sum_p = 0;
    for num = s:e
        if isprime(num)
            sum_p = sum_p + num;
        end
    end

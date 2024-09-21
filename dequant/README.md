# dequant
## host.cpp
    - 加三個buffer
        - qweight: dataPrepare_in4_weight
        - qzeros: dataPrepare_in4
        - scales: dataPrepare
## host.hpp
    - 加以上三個dataPrepare function，include可能要對一下，但我應該只有多加ap_int(應該@@)
## kernel
    - 可以看你要加哪，應該會取代掉pack那邊，但是最後一行放進去out的地方可能要測一下，輸出成一維的時候是對的
    - 三個多的AXI那邊
The Interface Board and the DUT (U575ZI) will likely use the same ID: 0x0483: 0x374E
This will be an issue if you wantto use the new ML Commons Runner.
This is not an issue in power measurements mode (the DUT is not addressed directly by the runner but from the interface board in pass through mode) 
In Performance measurement mode the runner will try to treat the DUT as an interface board, this can be resolved using the check_name attribute
but will not work if the runner cannot run a check name (at unexpected speed for example).

IF you do not need the interface board (AD/VWW/KWS/IC) simply use a devices yaml file that does not have any inteface board referenced.
IF you neeed the runner to address both the interface board and the dut you can use a separate USB adapter (FTDI or equivalent)
and use this instead of the VCP.
   You will need to find corresponding TX/RX pins on the nucleo Board located in the CN12 connector (will need to solder those connectors)
   Rx (PA10) is Pin 33 
   Tx (PA9) is Pin 21
   Find the VID/PID of your usb serial adapter (from the device manager in windows) and use that in the devices yaml file (dut description)

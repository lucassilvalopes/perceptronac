# Python program to implement Cohen Sutherland algorithm
# for line clipping.
# 
# https://stackoverflow.com/questions/3746274/line-intersection-with-aabb-rectangle
# https://www.geeksforgeeks.org/line-clipping-set-1-cohen-sutherland-algorithm/


# Defining region codes
INSIDE = 0 # 0000
LEFT = 1 # 0001
RIGHT = 2 # 0010
BOTTOM = 4 # 0100
TOP = 8 # 1000


# Function to compute region code for a point(x, y)
def computeCode(x_min, x_max, y_min, y_max, x, y):
	code = INSIDE
	if x < x_min: # to the left of rectangle
		code |= LEFT
	elif x > x_max: # to the right of rectangle
		code |= RIGHT
	if y < y_min: # below the rectangle
		code |= BOTTOM
	elif y > y_max: # above the rectangle
		code |= TOP
	return code


# Defining x_max, y_max and x_min, y_min for rectangle
# Since diagonal points are enough to define a rectangle
# Implementing Cohen-Sutherland algorithm
# Clipping a line from P1 = (x1, y1) to P2 = (x2, y2)
# Implementing Cohen-Sutherland algorithm
# Clipping a line from P1 = (x1, y1) to P2 = (x2, y2)
def cohenSutherlandClip(x_min, x_max, y_min, y_max, x1, y1, x2, y2):

	# Compute region codes for P1, P2
	code1 = computeCode(x_min, x_max, y_min, y_max, x1, y1)
	code2 = computeCode(x_min, x_max, y_min, y_max, x2, y2)
	accept = False

	while True:

		# If both endpoints lie within rectangle
		if code1 == 0 and code2 == 0:
			accept = True
			break

		# If both endpoints are outside rectangle
		elif (code1 & code2) != 0:
			break

		# Some segment lies within the rectangle
		else:

			# Line needs clipping
			# At least one of the points is outside,
			# select it
			x = 1.0
			y = 1.0
			if code1 != 0:
				code_out = code1
			else:
				code_out = code2

			# Find intersection point
			# using formulas y = y1 + slope * (x - x1),
			# x = x1 + (1 / slope) * (y - y1)
			if code_out & TOP:
				# Point is above the clip rectangle
				x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
				y = y_max
			elif code_out & BOTTOM:
				# Point is below the clip rectangle
				x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
				y = y_min
			elif code_out & RIGHT:
				# Point is to the right of the clip rectangle
				y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
				x = x_max
			elif code_out & LEFT:
				# Point is to the left of the clip rectangle
				y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
				x = x_min

			# Now intersection point (x, y) is found
			# We replace point outside clipping rectangle
			# by intersection point
			if code_out == code1:
				x1 = x
				y1 = y
				code1 = computeCode(x_min, x_max, y_min, y_max, x1, y1)
			else:
				x2 = x
				y2 = y
				code2 = computeCode(x_min, x_max, y_min, y_max, x2, y2)

	if accept:
		# print("Line accepted from %.2f, %.2f to %.2f, %.2f" % (x1, y1, x2, y2))

		# Here the user can add code to display the rectangle
		# along with the accepted (portion of) lines
        
		function_output = (x1, y1, x2, y2)

	else:
		# print("Line rejected")

		function_output = (None,None,None,None)

	return function_output

# Driver script

if __name__ == "__main__":

	# First line segment
	# P11 = (5, 5), P12 = (7, 7)
	cohenSutherlandClip(4.0,10.0,4.0,8.0,5, 5, 7, 7)

	# Second line segment
	# P21 = (7, 9), P22 = (11, 4)
	cohenSutherlandClip(4.0,10.0,4.0,8.0,7, 9, 11, 4)

	# Third line segment
	# P31 = (1, 5), P32 = (4, 1)
	cohenSutherlandClip(4.0,10.0,4.0,8.0,1, 5, 4, 1)

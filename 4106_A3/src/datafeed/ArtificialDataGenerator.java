package datafeed;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Random;

import classifier.ADSamplePoint;

/*  Generates the artificial dataset
 * 	Classes: 4
 *  Features: 10
 *  Note that the tree structure is only known to the ArtificialDataGenerator class
 *  which will generate the data and write it to a database.
 *  The classifiers will not know it.
 * */
public class ArtificialDataGenerator{

	static Connection database;
	static Statement stat;
	
	static final int NUMSAMPLES = 2000;
	
	
	/* Tree structure:
	 *               x5
	 *     +---------+--------+
	 *     |         |        |
	 *     x10       x9       x2
	 *     |         |        |
	 *     |         |   +----+----+  
	 *     x4        x8  |    |    |
	 *               |   |    |    |
	 *               x7  x1   x3   x6
	 *
	 *    +------------------+------------------+------------------+------------------+
	 *    | w1               | w2               | w3               | w4               |
	 *    +------------------+------------------+------------------+------------------+
	 *    | x5=0: 0.693      | x5=0: 0.259      | x5=0: 0.452      | x5=0: 0.854      |
	 *    +------------------+------------------+------------------+------------------+        
	 * 	  |x10=0|x5=0: 0.431 |x10=0|x5=0: 0.021 |x10=0|x5=0: 0.255 |x10=0|x5=0: 0.615 |		
	 * 	  |x10=0|x5=1: 0.103 |x10=0|x5=1: 0.463 |x10=0|x5=1: 0.577 |x10=0|x5=1: 0.108 |	
	 * 	  +------------------+------------------+------------------+------------------+
	 * 	  | x9=0|x5=0: 0.352 | x9=0|x5=0: 0.332 | x9=0|x5=0: 0.710 | x9=0|x5=0: 0.108 |		
	 * 	  | x9=0|x5=1: 0.138 | x9=0|x5=1: 0.841 | x9=0|x5=1: 0.892 | x9=0|x5=1: 0.437 |	 
	 *    +------------------+------------------+------------------+------------------+
	 *    | x2=0|x5=0: 0.735 | x2=0|x5=0: 0.662 | x2=0|x5=0: 0.704 | x2=0|x5=0: 0.801 |		
	 * 	  | x2=0|x5=1: 0.312 | x2=0|x5=1: 0.101 | x2=0|x5=1: 0.161 | x2=0|x5=1: 0.468 |	    
	 *    +------------------+------------------+------------------+------------------+
	 *    |x4=0|x10=0: 0.856 |x4=0|x10=0: 0.233 |x4=0|x10=0: 0.182 |x4=0|x10=0: 0.260 |		
	 * 	  |x4=0|x10=1: 0.291 |x4=0|x10=1: 0.661 |x4=0|x10=1: 0.983 |x4=0|x10=1: 0.296 |	    
	 *    +------------------+------------------+------------------+------------------+ 
	 *    | x8=0|x9=0: 0.580 | x8=0|x9=0: 0.699 | x8=0|x9=0: 0.362 | x8=0|x9=0: 0.048 |		
	 * 	  | x8=0|x9=1: 0.067 | x8=0|x9=1: 0.505 | x8=0|x9=1: 0.024 | x8=0|x9=1: 0.855 |	    
	 *    +------------------+------------------+------------------+------------------+
	 *    | x1=0|x2=0: 0.908 | x1=0|x2=0: 0.957 | x1=0|x2=0: 0.570 | x1=0|x2=0: 0.528 |		
	 * 	  | x1=0|x2=1: 0.501 | x1=0|x2=1: 0.829 | x1=0|x2=1: 0.135 | x1=0|x2=1: 0.946 |	    
	 *    +------------------+------------------+------------------+------------------+
	 *    | x3=0|x2=0: 0.040 | x3=0|x2=0: 0.215 | x3=0|x2=0: 0.419 | x3=0|x2=0: 0.829 |		
	 * 	  | x3=0|x2=1: 0.319 | x3=0|x2=1: 0.632 | x3=0|x2=1: 0.617 | x3=0|x2=1: 0.094 |	    
	 *    +------------------+------------------+------------------+------------------+
	 *    | x6=0|x2=0: 0.048 | x6=0|x2=0: 0.359 | x6=0|x2=0: 0.888 | x6=0|x2=0: 0.648 |		
	 * 	  | x6=0|x2=1: 0.768 | x6=0|x2=1: 0.198 | x6=0|x2=1: 0.293 | x6=0|x2=1: 0.848 |	    
	 *    +------------------+------------------+------------------+------------------+
	 *    | x7=0|x8=0: 0.392 | x7=0|x8=0: 0.941 | x7=0|x8=0: 0.341 | x7=0|x8=0: 0.662 |		
	 * 	  | x7=0|x8=1: 0.832 | x7=0|x8=1: 0.297 | x7=0|x8=1: 0.251 | x7=0|x8=1: 0.467 |	    
	 *    +------------------+------------------+------------------+------------------+ 
	 * 
	 * 
	 * */
	
	
	
	
	
	public static void main(String[] args) {
		/////////////////// Set up database
		//Connect to database
		
		int[] x = new int[10];
		long seed;
		
		try {
	
			//direct java to the sqlite-jdbc driver jar code
			// load the sqlite-JDBC driver using the current class loader
			Class.forName("org.sqlite.JDBC");
			System.out.println("Open Database Connection");
	
			//HARD CODED DATABASE NAME:
			database = DriverManager.getConnection("jdbc:sqlite:a3");
		     //create a statement object which will be used to relay a
		     //sql query to the database
			stat = database.createStatement();
			database.setAutoCommit(false);
	
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		//go do some stuff with the DB
		/*
		//generate class w1
				for(int i = 0; i < NUMSAMPLES; ++i){
					seed = System.nanoTime();
					x[5] = (new Random(seed).nextFloat() < 0.693) ? 0:1;
					//x10|x5
					if(x[5] == 0) x[0] = (new Random(seed).nextFloat() < 0.431) ? 0:1;
					else x[0] = (new Random(seed).nextFloat() < 0.103) ? 0:1;
					//x9|x5
					if(x[5] == 0) x[9] = (new Random(seed).nextFloat() < 0.352) ? 0:1;
					else x[9] = (new Random(seed).nextFloat() < 0.138) ? 0:1;
					//x2|x5
					if(x[5] == 0) x[2] = (new Random(seed).nextFloat() < 0.735) ? 0:1;
					else x[2] = (new Random(seed).nextFloat() < 0.312) ? 0:1;
					//x4|x10
					if(x[0] == 0) x[4] = (new Random(seed).nextFloat() < 0.856) ? 0:1;
					else x[4] = (new Random(seed).nextFloat() < 0.291) ? 0:1;
					//x8|x9
					if(x[9] == 0) x[8] = (new Random(seed).nextFloat() < 0.580) ? 0:1;
					else x[8] = (new Random(seed).nextFloat() < 0.067) ? 0:1;
					//x1|x2
					if(x[2] == 0) x[1] = (new Random(seed).nextFloat() < 0.908) ? 0:1;
					else x[1] = (new Random(seed).nextFloat() < 0.501) ? 0:1;
					//x3|x2
					if(x[2] == 0) x[3] = (new Random(seed).nextFloat() < 0.040) ? 0:1;
					else x[3] = (new Random(seed).nextFloat() < 0.319) ? 0:1;
					//x6|x2
					if(x[2] == 0) x[6] = (new Random(seed).nextFloat() < 0.048) ? 0:1;
					else x[6] = (new Random(seed).nextFloat() < 0.768) ? 0:1;
					//x7|x8
					if(x[8] == 0) x[7] = (new Random(seed).nextFloat() < 0.392) ? 0:1;
					else x[7] = (new Random(seed).nextFloat() < 0.832) ? 0:1;
					
					try{
						String sqlQueryString = "INSERT INTO artificialData(class, w1,w2,w3,w4,w5,w6,w7,w8,w9,w10) VALUES("
								+ 1 + "," + 
								+ x[1] + "," + 
								+ x[2] + "," + 
								+ x[3] + "," + 
								+ x[4] + "," + 
								+ x[5] + "," + 
								+ x[6] + "," + 
								+ x[7] + "," + 
								+ x[8] + "," + 
								+ x[9] + "," + 
								+ x[0] + ");" ;
						System.out.println("Class - 1 -- Data point " + i + ":  " + sqlQueryString);
						
						stat.executeUpdate(sqlQueryString);
					}
					catch(SQLException e){
						e.printStackTrace();			
					}
				}
				
				//generate class w2
				for(int i = 0; i < NUMSAMPLES; ++i){
					seed = System.nanoTime();
					x[5] = (new Random(seed).nextFloat() < 0.259) ? 0:1;
					//x10|x5
					if(x[5] == 0) x[0] = (new Random(seed).nextFloat() < 0.021) ? 0:1;
					else x[0] = (new Random(seed).nextFloat() < 0.463) ? 0:1;
					//x9|x5
					if(x[5] == 0) x[9] = (new Random(seed).nextFloat() < 0.332) ? 0:1;
					else x[9] = (new Random(seed).nextFloat() < 0.841) ? 0:1;
					//x2|x5
					if(x[5] == 0) x[2] = (new Random(seed).nextFloat() < 0.662) ? 0:1;
					else x[2] = (new Random(seed).nextFloat() < 0.101) ? 0:1;
					//x4|x10
					if(x[0] == 0) x[4] = (new Random(seed).nextFloat() < 0.233) ? 0:1;
					else x[4] = (new Random(seed).nextFloat() < 0.661) ? 0:1;
					//x8|x9
					if(x[9] == 0) x[8] = (new Random(seed).nextFloat() < 0.699 ) ? 0:1;
					else x[8] = (new Random(seed).nextFloat() < 0.505) ? 0:1;
					//x1|x2
					if(x[2] == 0) x[1] = (new Random(seed).nextFloat() < 0.957) ? 0:1;
					else x[1] = (new Random(seed).nextFloat() < 0.829) ? 0:1;
					//x3|x2
					if(x[2] == 0) x[3] = (new Random(seed).nextFloat() < 0.215) ? 0:1;
					else x[3] = (new Random(seed).nextFloat() < 0.032) ? 0:1;
					//x6|x2
					if(x[2] == 0) x[6] = (new Random(seed).nextFloat() < 0.359) ? 0:1;
					else x[6] = (new Random(seed).nextFloat() < 0.198) ? 0:1;
					//x7|x8
					if(x[8] == 0) x[7] = (new Random(seed).nextFloat() < 0.941) ? 0:1;
					else x[7] = (new Random(seed).nextFloat() < 0.297) ? 0:1;
					
					try{
						String sqlQueryString = "INSERT INTO artificialData(class, w1,w2,w3,w4,w5,w6,w7,w8,w9,w10) VALUES("
								+ 2 + "," + 
								+ x[1] + "," + 
								+ x[2] + "," + 
								+ x[3] + "," + 
								+ x[4] + "," + 
								+ x[5] + "," + 
								+ x[6] + "," + 
								+ x[7] + "," + 
								+ x[8] + "," + 
								+ x[9] + "," + 
								+ x[0] + ");" ;
						System.out.println("Class - 2 -- Data point " + i + ":  " + sqlQueryString);
						
						stat.executeUpdate(sqlQueryString);
					}
					catch(SQLException e){
						e.printStackTrace();			
					}
				}
				
				//generate class w3
				for(int i = 0; i < NUMSAMPLES; ++i){
					seed = System.nanoTime();
					x[5] = (new Random(seed).nextFloat() < 0.452) ? 0:1;
					//x10|x5
					if(x[5] == 0) x[0] = (new Random(seed).nextFloat() < 0.255) ? 0:1;
					else x[0] = (new Random(seed).nextFloat() < 0.577) ? 0:1;
					//x9|x5
					if(x[5] == 0) x[9] = (new Random(seed).nextFloat() < 0.710) ? 0:1;
					else x[9] = (new Random(seed).nextFloat() < 0.892) ? 0:1;
					//x2|x5
					if(x[5] == 0) x[2] = (new Random(seed).nextFloat() < 0.704) ? 0:1;
					else x[2] = (new Random(seed).nextFloat() < 0.161) ? 0:1;
					//x4|x10
					if(x[0] == 0) x[4] = (new Random(seed).nextFloat() < 0.182) ? 0:1;
					else x[4] = (new Random(seed).nextFloat() < 0.983) ? 0:1;
					//x8|x9
					if(x[9] == 0) x[8] = (new Random(seed).nextFloat() < 0.362) ? 0:1;
					else x[8] = (new Random(seed).nextFloat() < 0.024) ? 0:1;
					//x1|x2
					if(x[2] == 0) x[1] = (new Random(seed).nextFloat() < 0.570) ? 0:1;
					else x[1] = (new Random(seed).nextFloat() < 0.135) ? 0:1;
					//x3|x2
					if(x[2] == 0) x[3] = (new Random(seed).nextFloat() < 0.419) ? 0:1;
					else x[3] = (new Random(seed).nextFloat() < 0.617) ? 0:1;
					//x6|x2
					if(x[2] == 0) x[6] = (new Random(seed).nextFloat() < 0.888) ? 0:1;
					else x[6] = (new Random(seed).nextFloat() < 0.293) ? 0:1;
					//x7|x8
					if(x[8] == 0) x[7] = (new Random(seed).nextFloat() < 0.341) ? 0:1;
					else x[7] = (new Random(seed).nextFloat() < 0.251) ? 0:1;
					
					
					*/
				RNG gen = new RNG();
				ArrayList<ADSamplePoint> data = gen.get();
		
				for(ADSamplePoint pt: data){
					try{
						String sqlQueryString = "INSERT INTO artificialData(class, w1,w2,w3,w4,w5,w6,w7,w8,w9,w10) VALUES("
								+ pt.theClass + "," + 
								+ pt.features[1] + "," + 
								+ pt.features[2] + "," + 
								+ pt.features[3] + "," + 
								+ pt.features[4] + "," + 
								+ pt.features[5] + "," + 
								+ pt.features[6] + "," + 
								+ pt.features[7] + "," + 
								+ pt.features[8] + "," + 
								+ pt.features[9] + "," + 
								+ pt.features[0] + ");" ;
						System.out.println("Class - 3 -- Data point " + pt + ":  " + sqlQueryString);
						
						stat.executeUpdate(sqlQueryString);
					}
					catch(SQLException e){
						e.printStackTrace();			
					}
				}
				
	/*
				//generate class w4
				for(int i = 0; i < NUMSAMPLES; ++i){
					seed = System.nanoTime();
					x[5] = (new Random(seed).nextFloat() < 0.854) ? 0:1;
					//x10|x5
					if(x[5] == 0) x[0] = (new Random(seed).nextFloat() < 0.615) ? 0:1;
					else x[0] = (new Random(seed).nextFloat() < 0.108) ? 0:1;
					//x9|x5
					if(x[5] == 0) x[9] = (new Random(seed).nextFloat() < 0.108) ? 0:1;
					else x[9] = (new Random(seed).nextFloat() < 0.437) ? 0:1;
					//x2|x5
					if(x[5] == 0) x[2] = (new Random(seed).nextFloat() < 0.801) ? 0:1;
					else x[2] = (new Random(seed).nextFloat() < 0.468) ? 0:1;
					//x4|x10
					if(x[0] == 0) x[4] = (new Random(seed).nextFloat() < 0.260) ? 0:1;
					else x[4] = (new Random(seed).nextFloat() < 0.296) ? 0:1;
					//x8|x9
					if(x[9] == 0) x[8] = (new Random(seed).nextFloat() < 0.048) ? 0:1;
					else x[8] = (new Random(seed).nextFloat() < 0.855) ? 0:1;
					//x1|x2
					if(x[2] == 0) x[1] = (new Random(seed).nextFloat() < 0.528) ? 0:1;
					else x[1] = (new Random(seed).nextFloat() < 0.946) ? 0:1;
					//x3|x2
					if(x[2] == 0) x[3] = (new Random(seed).nextFloat() < 0.829) ? 0:1;
					else x[3] = (new Random(seed).nextFloat() < 0.694) ? 0:1;
					//x6|x2
					if(x[2] == 0) x[6] = (new Random(seed).nextFloat() < 0.648) ? 0:1;
					else x[6] = (new Random(seed).nextFloat() < 0.848) ? 0:1;
					//x7|x8
					if(x[8] == 0) x[7] = (new Random(seed).nextFloat() < 0.662) ? 0:1;
					else x[7] = (new Random(seed).nextFloat() < 0.467) ? 0:1;
					
					try{
						String sqlQueryString = "INSERT INTO artificialData(class, w1,w2,w3,w4,w5,w6,w7,w8,w9,w10) VALUES("
								+ 4 + "," + 
								+ x[1] + "," + 
								+ x[2] + "," + 
								+ x[3] + "," + 
								+ x[4] + "," + 
								+ x[5] + "," + 
								+ x[6] + "," + 
								+ x[7] + "," + 
								+ x[8] + "," + 
								+ x[9] + "," + 
								+ x[0] + ");" ;
						System.out.println("Class - 4 -- Data point " + i + ":  " + sqlQueryString);
						
						stat.executeUpdate(sqlQueryString);
					}
					catch(SQLException e){
						e.printStackTrace();			
					}
				}
				// */
		

		try {
			database.setAutoCommit(true);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//close the connection
		try {
			System.out.println("Closing Database Connection");
			database.close();
		} catch (SQLException e1) {
			e1.printStackTrace();
		}
	}	
}




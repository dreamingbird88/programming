// ================================================================= //
阅读记录: 2013.10.11 
// ================================================================= //
http://www.cnblogs.com/whyandinside/archive/2012/10/26/2740612.html
OOD的面试的时要特别注意Clarify the ambiguity，以避免你的设计既可以满足需求，又不会over design。
不要自己假设，在考虑应不应该使用某种设计时，问面试官以确定各种条件是否符合

OOD 设计一个魔方，实现一个rotation method

电梯OOD，实现最短等待时间{
	class Cab{
		Capacity
		LoadWeight
		Floor
		Status
		serveRequest();
	}
	
	Request{
		floor, direction, weight;
	}
	
	class CenterControl{
		vector<Cab> cabs;
		AssignedCab(Request);
	}
}

Parking Lot{
	class car{
		int size, brand, model, color, etc.
	}
	
	class ParkingSpace{
		position, 
		car,
		enum status {vacant, notAvailable},
		subcalss handicap, compact, large etc
	}
	
	class OutletInLet{
		position
		status 
		subclass Entrance, Exit
	}
	
	class ParkingLot{
		vector<ParkingSpace> spaceArray;
		vector<Entrance> spaceArray;
		vector<Exit> spaceArray;
		
		int Size;
		int AvailableSpaceNum;
		HandleService(ParkingService){
		}
		
	}
	
	class ParkingService{
		status = {Rejected, Served, Parking, Entering, Exiting}
		car, 
		ParkingSpace,
		Extrance,
		Exit,
		EnteringTime, ExitingTime
		
	}

}

Restarent Reservation System{
	class Person{
		age, skins, 
		subclass Cook, Servant, Guest{Table, status}
	}
	
	class Service{
	}
	
	class Table{
		size, position, status
		vector<Service>
	}
	
	class Request{
		Person p; 
		subclass ServantRequest, GuestRequest, CookRequest
	}
	
	Restrant{
		vector<Table> tables;
		vector<Servant> servants;
		vector<Cook> cooks;
		vector<Material>
		
		HandleRequest(Person)
	}
}

http://www.cnblogs.com/quanfengnan/archive/2012/10/22/2733922.html

23.    设计一个类amazon商品的page,包括的内容有product info, reviews, recommendations (like ppl who buy this also buy),要求user customized.
24.    设计一个youtube的type ahead search bar,涉及到数据结构，distributed hashtable和估算内存。